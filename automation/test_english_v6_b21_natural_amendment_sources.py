"""Offline source-amendment policy, provenance, budget and routing tests."""
from __future__ import annotations
import ast
import base64
import copy
import io
import json
from pathlib import Path
import re
import sys
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_natural_amendment_source as source
import english_v6_b21_natural_amendment_source_ancestry as lineage
import english_v6_b21_natural_amendment_natural_sources as natural
import english_v6_b21_natural_amendment_source_producer as producer
import english_v6_b21_natural_amendment_transport as transport

HERE = Path(__file__).resolve().parent


def functions(name):
    return {node.name: node for node in ast.parse((HERE / name).read_bytes()).body if isinstance(node, ast.FunctionDef)}


def remove_natural_journals(function):
    class Remove(ast.NodeTransformer):
        def visit_Assign(self, node):
            return None if any(isinstance(target, ast.Name) and target.id == "decode_dir" for target in node.targets) else self.generic_visit(node)
        def visit_Expr(self, node):
            text = ast.unparse(node)
            if text.startswith("decode_dir.mkdir(") or text.startswith("qa.record(decode_dir") or "all24 original native decode starts and returns retained" in text:
                return None
            return self.generic_visit(node)
    return Remove().visit(copy.deepcopy(function))


def execution():
    return dict(run_id="123", run_head="b"*40, run_attempt=1, run_branch=lineage.BRANCH, prepared_parent="a"*40)


def registration():
    return dict(execution(), source_manifest_sha256="c"*64, source_rows=120, max_calls=120)


def budget_fixture(provider="qwen"):
    document = lineage.observer_document(provider, registration())
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("OBSERVER_CONSUMPTION.json", json.dumps(document))
    raw = stream.getvalue()
    artifact = dict(id=456, name=document["marker_name"], expired=False, size_in_bytes=len(raw),
        digest="sha256:" + lineage.digest(raw), workflow_run=dict(id=123, head_sha="b"*40))
    return dict(artifact=artifact, original_zip_base64=base64.b64encode(raw).decode(), receipt=document)


class AmendmentSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bound = lineage.selection.verify_registration()

    def test_inert_registration_stops_before_source_access(self):
        with mock.patch.object(lineage.selection, "verify_registration", return_value=dict(bound=False, primary_roster=None, replication_roster=None)):
            with self.assertRaisesRegex(Exception, "committed exact"):
                lineage.registered()

    def test_amended_reserved_readers_are_visible_to_unchanged_scorer(self):
        registered = copy.deepcopy(self.bound)
        if not registered["bound"]:
            base = lineage.frozen.read_plan()
            registered.update(bound=True, primary_design=dict(base["primary"]["natural"], speakers=["100000"+str(i) for i in range(12)]),
                replication_design=dict(base["replication"]["natural"], speakers=["200000"+str(i) for i in range(12)]))
        with mock.patch.object(lineage.selection, "verify_registration", return_value=registered):
            actual = lineage.read_plan()
        old = lineage.frozen.read_plan()
        self.assertEqual(actual["source_gate"], old["source_gate"])
        self.assertEqual(actual["quality_criteria"], old["quality_criteria"])
        self.assertEqual(actual["replication"]["natural"]["speakers"], registered["replication_design"]["speakers"])
        self.assertEqual(source.ASSESSMENT_SHA, lineage.frozen.ASSESSMENT_SHA)

    def test_exact_committed_roster_rejects_runtime_substitution(self):
        rows = [dict(source_id="one")]
        with mock.patch.object(natural.original, "rank_metadata", return_value=rows), mock.patch.object(lineage, "registered", return_value=dict(primary_roster=rows)):
            self.assertEqual(natural.rank_metadata({}, {}), rows)
        with mock.patch.object(natural.original, "rank_metadata", return_value=[dict(source_id="replacement")]), mock.patch.object(lineage, "registered", return_value=dict(primary_roster=rows)):
            with self.assertRaisesRegex(Exception, "exact committed24"):
                natural.rank_metadata({}, {})

    def test_bound_roster_preserves_exact_original_filter_and_slots(self):
        if not self.bound["bound"]:
            return
        for panel in ("primary", "replication"):
            design, rows = self.bound[panel + "_design"], self.bound[panel + "_roster"]
            self.assertEqual(len(rows), 24)
            self.assertEqual([row["speaker"] for row in rows], [speaker for speaker in design["speakers"] for _ in range(2)])
            self.assertEqual(design["utterance_selection"], lineage.frozen.read_plan()[panel]["natural"]["utterance_selection"])
            self.assertFalse(design["opened"])
        self.assertFalse(set(self.bound["primary_design"]["speakers"]) & set(self.bound["replication_design"]["speakers"]))

    def test_admission_science_ast_is_identical(self):
        old = functions("english_v6_b21_fixed_mirror_source.py")
        new = functions("english_v6_b21_natural_amendment_source.py")
        for name in ("screen_document", "verify_screen_contract", "screen"):
            self.assertEqual(ast.dump(old[name]), ast.dump(new[name]), name)

    def test_natural_audio_science_only_adds_decode_receipts(self):
        old = functions("english_v6_b21_natural_sources.py")
        new = functions("english_v6_b21_natural_amendment_natural_sources.py")
        self.assertEqual(ast.dump(old["verify_corpus"]), ast.dump(new["verify_corpus"]))
        for name in ("extract", "read_source"):
            self.assertEqual(ast.dump(old[name]), ast.dump(remove_natural_journals(new[name])), name)
        self.assertIs(natural.flac_header, natural.original.flac_header)
        self.assertIs(natural.whole_wav_facts, natural.original.whole_wav_facts)

    def test_transport_science_ast_unchanged(self):
        old = functions("english_v6_b21_fixed_mirror_transport.py")
        new = functions("english_v6_b21_natural_amendment_transport.py")
        self.assertEqual(ast.dump(old["download"]), ast.dump(new["download"]))
        self.assertEqual(transport.plan()["parallel_ranges"], 8)
        self.assertEqual(transport.plan()["range_retries"], 0)

    def test_no_synthetic_generation_entrypoint(self):
        raw = (HERE / "english_v6_b21_natural_amendment_source.py").read_text()
        tree = ast.parse(raw)
        calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        self.assertFalse(any("render_panel" in name or "synthesize" in name or "PiperVoice" in name for name in calls))
        self.assertNotIn('add_parser("generate")', raw)
        self.assertIn('"ASSEMBLY_STARTED.json"', raw)
        self.assertNotIn('qa.record(out / "STARTED.json"', raw)
        self.assertNotIn('qa.record(out / "SYNTHETIC_MANIFEST.json"', raw)

    def test_guard_rejects_retry_and_foreign_parent(self):
        identity = execution()
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=lineage.REPO, GITHUB_REF=lineage.BRANCH,
            GITHUB_EVENT_NAME="push", GITHUB_RUN_ATTEMPT="1", GITHUB_RUN_ID=identity["run_id"], GITHUB_SHA=identity["run_head"])
        event = dict(before=identity["prepared_parent"], after=identity["run_head"], ref=lineage.BRANCH, forced=False,
            deleted=False, repository=dict(full_name=lineage.REPO, private=False))
        with mock.patch.object(lineage, "read_plan"), mock.patch.object(lineage, "registered"), mock.patch.object(lineage.origin, "verify"):
            self.assertEqual(lineage.execution_guard(identity["prepared_parent"], env, event, identity["prepared_parent"]), identity)
            with self.assertRaises(Exception):
                lineage.execution_guard(identity["prepared_parent"], dict(env, GITHUB_RUN_ATTEMPT="2"), event, identity["prepared_parent"])
            with self.assertRaises(Exception):
                lineage.execution_guard(identity["prepared_parent"], env, event, "d"*40)

    def test_observer_uploaded_budget_must_match_exact_provider_and_source(self):
        value = budget_fixture()
        lineage.validate_observer_budget(value, "qwen", registration())
        for forged in (dict(value, artifact=dict(value["artifact"], expired=True)),
                       dict(value, artifact=dict(value["artifact"], digest="sha256:" + "0"*64)),
                       dict(value, artifact=dict(value["artifact"], workflow_run=dict(id=999, head_sha="b"*40)))):
            with self.assertRaises(Exception):
                lineage.validate_observer_budget(forged, "qwen", registration())
        with self.assertRaises(Exception):
            lineage.validate_observer_budget(value, "whisper", registration())
        with self.assertRaises(Exception):
            lineage.validate_observer_budget(value, "qwen", dict(registration(), source_manifest_sha256="f"*64))

    def test_metadata_refresh_rejects_rerun_missing_or_expired_artifact(self):
        if not self.bound["bound"]:
            return
        registration_value = self.bound
        identity = registration_value["metadata_execution"]
        public = dict(full_name=lineage.REPO, private=False)
        run = dict(id=int(identity["run_id"]), head_sha=identity["run_head"], run_attempt=1, event="push",
            head_branch=identity["run_branch"].removeprefix("refs/heads/"),
            path=".github/workflows/english-v6-b21-natural-metadata-once.yml", status="completed", conclusion="success",
            repository=public, head_repository=public)
        artifacts = []
        for key in ("metadata_artifact_pin", "metadata_producer_pin", "metadata_consumed_pin"):
            pin = registration_value[key]
            artifacts.append(dict(id=pin["id"], size_in_bytes=pin["bytes"], digest="sha256:"+pin["sha256"], expired=False,
                workflow_run=dict(id=pin["run_id"], head_sha=pin["run_head"])))
        live = dict(run=run, artifacts=dict(total_count=3, artifacts=artifacts))
        with mock.patch.object(lineage, "registered", return_value=registration_value):
            lineage.validate_metadata_live(live)
            for changed in (dict(live, run=dict(run, run_attempt=2)), dict(live, artifacts=dict(total_count=2, artifacts=artifacts[:2]))):
                with self.assertRaises(Exception):
                    lineage.validate_metadata_live(changed)
            changed = copy.deepcopy(live)
            changed["artifacts"]["artifacts"][0]["expired"] = True
            with self.assertRaises(Exception):
                lineage.validate_metadata_live(changed)

    def test_workflow_consumes_each_budget_before_calls(self):
        template = (HERE / "english_v6_b21_natural_amendment_source_auth_workflow_template.yml").read_text()
        actual = (HERE.parent / lineage.WORKFLOW).read_text()
        token = "__B21_NATURAL_AMENDMENT_AUTH_MAIN_SHA__"
        parent = re.search(r"PREPARED_PARENT: '([^']+)'", actual).group(1)
        self.assertTrue(parent == token or re.fullmatch(r"[0-9a-f]{40}", parent))
        self.assertEqual(actual, template.replace(token, parent))
        self.assertEqual(template.count(token), 2)
        self.assertNotIn("piper-tts", template)
        self.assertNotIn("source.py generate", template)
        self.assertLess(template.index("name: english-v6-b21-natural-amendment-auth-source-consumed"), template.index("source.py assemble"))
        self.assertLess(template.index("name: english-v6-b21-natural-amendment-auth-observer-${{ matrix.provider }}-consumed"), template.index("name: Install one pinned native observer runtime"))
        self.assertIn("source.py consume-observer", template)
        self.assertIn("METADATA_LIVE.json", template)

    def test_producer_closure_has_no_mutable_scoring_inputs(self):
        python = producer.common.prior.static_closure(HERE, producer.ROOTS + producer.frozen_producer.ROOTS)
        self.assertFalse(any(("/english_v6_b21_" in name and name.endswith(("_primary_eval.py", "_replication_eval.py", "_engineering.py")))
                             or "/bind_b21" in name for name in python))
        resources = (producer.common.resource_paths(HERE.parent, python) | set(lineage.selection.resource_paths(HERE.parent))
                     | (set(lineage.selection.FROZEN_PATHS.values()) - python))
        self.assertFalse(any("/english_v6_b21_" in name and name.endswith(("_primary_inputs.json", "_replication_inputs.json", "_engineering_inputs.json")) for name in resources))
        self.assertIn("automation/english_v6_b21_natural_amendment_inputs.json", resources)
        self.assertTrue(set(lineage.selection.FROZEN_PATHS.values()) <= python | resources)
        self.assertTrue(all((HERE.parent / name).is_file() for name in python | resources))


if __name__ == "__main__":
    unittest.main()
