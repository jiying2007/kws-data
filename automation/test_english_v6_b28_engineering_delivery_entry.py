"""Offline B28 bridge refusal contracts; fictional checkout metadata only.

No fixture is a quality/engineering PASS, actual selected candidate, reviewed
pipeline commit, package, installation, or permission to execute delivery.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
definition = importlib.util.spec_from_file_location(
    "b28_delivery_entry_contract", HERE / "english_v6_b28_engineering_delivery_entry.py")
entry = importlib.util.module_from_spec(definition)
definition.loader.exec_module(entry)


def inert():
    return dict(schema=entry.SCHEMA, enabled=False, pipeline_repository="jiying2007/kws-pipeline",
                pipeline_commit=None, manifest_sha256=None, pipeline_files=None)


def fictional_checkout():
    # A structure fixture only. No file or commit with these identities exists.
    return dict(schema=entry.SCHEMA, enabled=True, pipeline_repository="jiying2007/kws-pipeline",
        pipeline_commit="a" * 40, manifest_sha256="b" * 64,
        pipeline_files={"research/english_v6_delivery/once_b28.py":
            dict(bytes=12, sha256="c" * 64, git_blob_sha1="d" * 40, mode="100644")})


class EntryContracts(unittest.TestCase):
    def test_registration_is_complete_or_explicitly_null(self):
        value = entry.contract(json.loads(entry.SPEC.read_bytes()))
        if value["enabled"] is False:
            self.assertEqual(value, inert())
        self.assertTrue({"numpy", "torch", "onnxruntime", "soundfile"}.isdisjoint(sys.modules))

    def test_exact_original_code_objects_and_globals_are_preserved(self):
        self.assertEqual(hashlib.sha256(entry.ORIGINAL.read_bytes()).hexdigest(), entry.ORIGINAL_SHA)
        self.assertIs(entry.contract.__code__, entry.original.contract.__code__)
        self.assertIs(entry.main.__code__, entry.original.main.__code__)
        self.assertEqual(entry.contract.__globals__["SCHEMA"], entry.SCHEMA)
        self.assertEqual(entry.contract.__globals__["SPEC"], entry.SPEC)
        self.assertEqual(entry.original.SCHEMA, "english-v6-b24r1-software-delivery-entry-v1")
        self.assertEqual(entry.original.SPEC.name, "english_v6_b24r1_delivery_inputs.json")
        self.assertIs(entry.original.contract.__globals__["contract"], entry.original.contract)

    def test_disabled_or_null_entry_cannot_smuggle_a_partial_selection(self):
        for key in ("pipeline_commit", "manifest_sha256", "pipeline_files"):
            changed = inert()
            changed[key] = "invented"
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry.contract(changed)
        with self.assertRaises(ValueError):
            entry.contract(dict(inert(), enabled=True))
        for value in (0, 1, "false", None):
            with self.subTest(enabled=value), self.assertRaises(ValueError):
                entry.contract(dict(inert(), enabled=value))

    def test_prior_schema_or_extra_evidence_summary_cannot_enable_bridge(self):
        for key, value in (("schema", "english-v6-b27-software-delivery-entry-v1"),
                           ("engineering_status", "PASS"), ("promotion_allowed", True),
                           ("candidate_pt_sha256", "e" * 64)):
            changed = inert()
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                entry.contract(changed)

    def test_branch_alias_output_injection_external_repository_and_empty_inventory_fail(self):
        fixture = fictional_checkout()
        entry.contract(fixture)
        for key, value in (("pipeline_commit", "main"), ("pipeline_commit", "latest"),
                           ("pipeline_commit", "a" * 40 + "\nselected=true"),
                           ("pipeline_repository", "other/repository"),
                           ("manifest_sha256", "B" * 64), ("pipeline_files", {})):
            changed = copy.deepcopy(fixture)
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                entry.contract(changed)

    def test_pipeline_members_require_complete_regular_git_bound_pins(self):
        fixture = fictional_checkout()
        name = next(iter(fixture["pipeline_files"]))
        for bad_name in ("/absolute.py", "../escape.py", "sub/../escape.py", "back\\slash.py"):
            changed = copy.deepcopy(fixture)
            changed["pipeline_files"] = {bad_name: changed["pipeline_files"][name]}
            with self.subTest(path=bad_name), self.assertRaises(ValueError):
                entry.contract(changed)
        for key, value in (("bytes", True), ("bytes", 0), ("bytes", 4_000_001),
                           ("sha256", "bad"), ("git_blob_sha1", "d" * 39),
                           ("mode", "120000")):
            changed = copy.deepcopy(fixture)
            changed["pipeline_files"][name][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                entry.contract(changed)
        changed = copy.deepcopy(fixture)
        changed["pipeline_files"][name].pop("git_blob_sha1")
        with self.assertRaises(ValueError):
            entry.contract(changed)

    def test_committed_check_reports_metadata_only_and_closed_pipeline_selection_refuses(self):
        command = [sys.executable, "-I", "-S", "-B", str(HERE / "english_v6_b28_engineering_delivery_entry.py")]
        checked = subprocess.run(command + ["check"], capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        value = entry.contract(json.loads(entry.SPEC.read_bytes()))
        self.assertEqual(json.loads(checked.stdout), dict(
            status="REVIEWED_PIPELINE_BOUND" if value["enabled"] else "INERT",
            model_forwards=0, packs_created=0, installation_started=False))
        if not value["enabled"]:
            refused = subprocess.run(command + ["pipeline-commit"], capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertEqual(refused.stdout, "")
            self.assertIn("INERT: no reviewed selected pipeline checkout", refused.stderr)

    def test_template_is_exactly_inert_or_only_two_reviewed_parent_replacements(self):
        template = (HERE / "english_v6_b28_engineering_delivery_workflow_template.yml").read_bytes()
        workflow = (HERE.parent / ".github/workflows/english-v6-b28-engineering-delivery-once.yml").read_bytes()
        token = b"__B28_REVIEWED_DELIVERY_PARENT__"
        self.assertEqual(template.count(token), 2)
        normalized = template.replace(b"B28", b"B27").replace(b"b28", b"b27")
        self.assertEqual(hashlib.sha256(normalized).hexdigest(),
                         "88e74aa01271b26611b78e107f122a1a052e733a17fa1808a332c0e5b12e9765")
        if workflow != template:
            parent = re.search(rb"PREPARED_PARENT: '([a-f0-9]{40})'", workflow)
            self.assertIsNotNone(parent)
            self.assertEqual(workflow, template.replace(token, parent.group(1)))
        text = template.decode()
        for guard in ("github.event_name == 'push'", "github.repository == 'jiying2007/kws-data'",
                      "github.event.repository.private == false", "github.run_attempt == 1", "needs: contracts",
                      "github.ref == 'refs/heads/research/en-v6-b28-engineering-delivery-once-20261125'",
                      "github.event.before == '__B28_REVIEWED_DELIVERY_PARENT__'"):
            self.assertIn(guard, text)
        for route in ("workflow_dispatch:", "workflow_run:", "schedule:", "secrets.",
                      "contents: write", "actions: write"):
            self.assertNotIn(route, text)
        self.assertIn("cancel-in-progress: false", text)

    def test_pipeline_handshake_and_retained_original_delivery_bytes_match(self):
        text = (HERE / "english_v6_b28_engineering_delivery_workflow_template.yml").read_text()
        self.assertIn("ref: '${{ steps.selected.outputs.pipeline_commit }}'", text)
        self.assertIn("GH_TOKEN: '${{ github.token }}'", text)
        self.assertIn("pipeline/research/english_v6_delivery/once_b28.py\" execute", text)
        self.assertIn('--data-root "$GITHUB_WORKSPACE/data" --prepared-parent "$PREPARED_PARENT"', text)
        self.assertIn('--output "$RUNNER_TEMP/b28-software-delivery"', text)
        self.assertIn("name: english-v6-b28-software-delivery-${{ github.run_id }}", text)
        self.assertIn("b28-software-delivery/heym43-B28-software-delivery.tar.gz", text)
        self.assertIn("b28-software-delivery/receipts/", text)
        self.assertIn("include-hidden-files: true", text)
        self.assertIn("if: always()", text)
        self.assertNotIn("once_b27.py", text)


if __name__ == "__main__":
    unittest.main()
