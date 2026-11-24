"""Sealed replica provenance and immutable scientific boundaries; no real calls."""
from __future__ import annotations
import ast
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_replication_source as source
import english_v6_b21_replication_source_ancestry as lineage
import english_v6_b21_replication_transport as transport
import english_v6_b21_natural_amendment_replication_natural_sources as natural


def identity():
    return dict(run_id="123", run_head="2" * 40, run_attempt=1,
                run_branch=lineage.BRANCH, prepared_parent="1" * 40)


def marker_fixture(document, member):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, json.dumps(document).encode())
    raw = stream.getvalue()
    return dict(artifact=dict(id=7, name=document["marker_name"], expired=False,
        workflow_run=dict(id=int(document["source_execution"]["run_id"]),
                          head_sha=document["source_execution"]["run_head"]),
        size_in_bytes=len(raw), digest="sha256:" + lineage.digest(raw)),
        receipt=document, original_zip_base64=base64.b64encode(raw).decode())


class AmendedReplication(unittest.TestCase):
    def test_raw_github_original_bytes_preserve_escape_sequences_without_decoding(self):
        raw = b"PK\x03\x04\x00\x1b[0m\xff\x80original\r\n"
        endpoint = "actions/jobs/123/logs"
        with patch.object(lineage.subprocess, "run", return_value=SimpleNamespace(stdout=raw)) as capture:
            self.assertIs(lineage.gh(endpoint), raw)
            capture.assert_called_once_with(
                ["gh", "api", "repos/" + lineage.REPO + "/" + endpoint, "--allow-escape-sequences"],
                stdout=lineage.subprocess.PIPE, stderr=lineage.subprocess.PIPE, check=True)

    def test_new_exact_reserve_and_old_reader_exclusion(self):
        value = lineage.registered()
        self.assertEqual(natural.plan(), value["replication_design"])
        self.assertEqual(natural.plan()["speakers"],
                         ["4195", "198", "60", "8098", "4018", "1926",
                          "6848", "6367", "5688", "298", "83", "248"])
        old = lineage.previous.read_plan()["replication"]["natural"]["speakers"]
        self.assertFalse(set(natural.plan()["speakers"]) & set(old))
        self.assertTrue(set(old) <= set(natural.plan()["excluded_canonical_speakers"]))
        self.assertFalse(set(natural.plan()["speakers"]) &
                         set(value["primary_design"]["speakers"]))
        root = Path(lineage.amended_primary.selection.__file__).resolve().parents[1]
        with zipfile.ZipFile(root / lineage.amended_primary.selection.EVIDENCE /
                             "metadata-original.zip") as archive:
            catalog = json.loads(archive.read("FULL_METADATA_CATALOG.json"))
        speakers = set(natural.plan()["speakers"])
        retained = dict(schema="english-v6-b21-replication-natural-original-metadata-v1",
            speakers_text=catalog["speakers_text"],
            transcripts=[{key: item[key] for key in ("path", "sha256", "text")}
                         for item in catalog["transcripts"]
                         if item["path"].split("/")[2] in speakers])
        self.assertEqual(natural.rank_metadata(retained, natural.plan()), value["replication_roster"])
        bad = copy.deepcopy(retained)
        selected_member = value["replication_roster"][0]["transcript_member"]
        bad["transcripts"] = [item for item in bad["transcripts"] if item["path"] != selected_member]
        with self.assertRaises(ValueError):
            natural.rank_metadata(bad, natural.plan())

    def test_both_original_ranker_files_remain_byte_frozen(self):
        rule, _, _ = lineage.amended_primary.selection.load_rule()
        root = Path(lineage.amended_primary.selection.__file__).resolve().parents[1]
        for role in ("primary", "replication"):
            path = lineage.amended_primary.selection.FROZEN_PATHS[role + "_natural_code_sha256"]
            self.assertEqual(lineage.digest((root / path).read_bytes()),
                             rule["frozen_inputs"][role + "_natural_code_sha256"])

    def test_synthetic_recipe_and_source_admission_functions_unchanged(self):
        expected = {
            "source_plan": "5531a8b64f060536a057b5713a68c89b7f085f6d143e986599138dd785f6f4ed",
            "validate_config": "95eab1418ffc637b8f831d470fc936cbf06d065ceffacf46e2ba0fb61cf4f97a",
            "verify_synthetic": "477f60790c3964585a15f9c8855a246ea3c4a1c9a41db5a0c3583f3e3a1b54f6",
            "screen_document": "c2bfe65a863ea1ca311506009536ec73edd7560c4795e6b5ec9073e492ad0bc6",
            "verify_screen_contract": "323b9e4f1fb4fffe922b9bcaaac6c8099b056375073a1c7e28f47fa850f6e5e7"}
        nodes = {node.name: node for node in ast.parse(Path(source.__file__).read_bytes()).body
                 if isinstance(node, ast.FunctionDef)}
        self.assertEqual({name: hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest()
                          for name in expected}, expected)
        self.assertEqual(source.source_plan(), source.primary.source_plan("replication"))
        self.assertEqual(len(source.source_plan()["rows"]), 96)

    def test_null_primary_lock_stops_every_scientific_or_corpus_entry(self):
        sealed = json.loads(lineage.OPENING.read_bytes())
        sealed.update(status="SEALED_UNTIL_ORIGINAL_PRIMARY_PASS",
                      primary_quality_lock=None, quality_readback_sha256=None)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            opening = root / "opening.json"
            opening.write_text(json.dumps(sealed))
            with patch.object(lineage, "OPENING", opening), \
                 patch.object(lineage, "gh", side_effect=AssertionError("no network")) as api, \
                 patch.object(source.B13, "render_panel", side_effect=AssertionError("no TTS")) as tts, \
                 patch.object(natural, "verify_corpus", side_effect=AssertionError("no corpus")) as corpus, \
                 patch.object(source.observers, "_load_phone", side_effect=AssertionError("no native")) as native:
                actions = [
                    lambda: lineage.authorize(root / "authorization", "1" * 40),
                    lambda: transport.download(root / "train-clean-100.tar.gz", root / "prior", "1" * 40),
                    lambda: source.generate(root / "out", root / "prior", root / "corpus", "1" * 40),
                    lambda: natural.extract(root / "source", root / "corpus", {}),
                    lambda: lineage.consume_observer("xlsr", root / "source", root / "budget", "1" * 40),
                    lambda: source.observe("xlsr", root / "source", root / "native", "1" * 40)]
                for action in actions:
                    with self.assertRaisesRegex(ValueError, "replication remains sealed"):
                        action()
                for mocked in (api, tts, corpus, native):
                    mocked.assert_not_called()
            self.assertEqual([path.name for path in root.iterdir()], ["opening.json"])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("soundfile", sys.modules)

    def test_original_source_marker_and_provider_markers_are_independent(self):
        markers = [lineage.marker_name()] + [lineage.marker_name(p) for p in lineage.PROVIDERS]
        self.assertEqual(len(set(markers)), 6)
        self.assertEqual(markers[0], "en-v6-b21-replication-panel-opened")
        self.assertFalse(set(markers) & {lineage.amended_primary.marker_name(p)
                                        for p in (None, *lineage.PROVIDERS)})
        self.assertEqual(lineage.SOURCE_BUDGET,
                         dict(maximum_tts_calls=96, maximum_natural_decodes=24,
                              maximum_native_observer_calls=600, new_kws_forwards=0, retry_count=0))

    def test_uploaded_budget_rejects_another_run_or_modified_receipt(self):
        expected = dict(marker_name=lineage.MARKER, source_execution=identity(),
                        source_budget=lineage.SOURCE_BUDGET)
        proof = marker_fixture(expected, "AUTHORIZATION.json")
        self.assertEqual(lineage.validate_uploaded_marker(proof, expected, "AUTHORIZATION.json"), expected)
        for mutation in ("run", "digest", "receipt"):
            bad = copy.deepcopy(proof)
            if mutation == "run":
                bad["artifact"]["workflow_run"]["id"] += 1
            elif mutation == "digest":
                bad["artifact"]["digest"] = "sha256:" + "0" * 64
            else:
                bad["receipt"]["source_budget"]["maximum_tts_calls"] += 1
            with self.assertRaises(ValueError, msg=mutation):
                lineage.validate_uploaded_marker(bad, expected, "AUTHORIZATION.json")

    def test_generation_entry_cannot_repeat_into_another_output(self):
        permission = dict(source_execution=identity())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "AUTHORIZATION.json").write_text("{}")
            with patch.object(lineage, "require_uploaded_opening", return_value=permission), \
                 patch.object(lineage, "amendment_binding", return_value={"fixture": True}):
                lineage.require_generation_permit(root, "1" * 40)
                with self.assertRaisesRegex(ValueError, "never retry"):
                    lineage.require_generation_permit(root, "1" * 40)
            self.assertTrue((root / "GENERATION_STARTED.json").is_file())

    def test_later_execution_or_foreign_provider_budget_is_rejected(self):
        current = identity()
        run = dict(id=123, head_sha=current["run_head"], run_attempt=1)
        def api(endpoint):
            if endpoint.startswith("actions/workflows/"):
                return json.dumps(dict(total_count=2, workflow_runs=[run, dict(run, id=124)])).encode()
            raise AssertionError("must stop before artifact access")
        with patch.object(lineage, "gh", side_effect=api):
            with self.assertRaisesRegex(ValueError, "one replication execution"):
                lineage.one_execution(current, False)
        with self.assertRaises(ValueError):
            lineage.marker_name("unregistered")

    def test_workflow_consumes_each_provider_before_model_installation(self):
        template = Path(source.__file__).with_name("english_v6_b21_replication_source_workflow_template.yml").read_text()
        self.assertEqual(template.count("__B21_REPLICATION_SOURCE_MAIN_SHA__"), 2)
        self.assertLess(template.index("source.py authorize "), template.index("replication_transport.py download "))
        self.assertLess(template.index("name: en-v6-b21-replication-panel-opened"),
                        template.index("replication_transport.py download "))
        self.assertLess(template.index(" consume-observer "), template.index("Install one pinned native observer runtime"))
        self.assertLess(template.index("name: english-v6-b21-replication-observer-"),
                        template.index("Save native return before validation"))
        self.assertNotIn("workflow_dispatch", template)


if __name__ == "__main__":
    unittest.main()
