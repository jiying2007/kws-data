"""No model imports or network: B21 primary boundary and consumption fixtures."""
import ast
import base64
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
definition = importlib.util.spec_from_file_location("b21_primary_contract", HERE / "english_v6_b21_fixed_mirror_primary_eval.py")
primary = importlib.util.module_from_spec(definition)
definition.loader.exec_module(primary)


def row(index, cohort="synthetic", positive=False):
    result = dict(id="row-" + str(index), cohort=cohort, observed_class=None,
                  requested_class=2 if positive else 0, pcm_sha256=f"{index:064x}",
                  wav_sha256=f"{index + 1000:064x}", samples=16000,
                  source_group_id="group-" + str(index), role="development")
    if cohort == "synthetic":
        result.update(source_speaker_id="p" + str(index), cell_id="mero-fast" if positive else "request")
    else:
        result.update(speaker=str(index), source_id=str(index) + "-a", transcript="A plain sentence")
    return result


def label(original, unknown=False):
    value = dict(original, observed_class=None if unknown else original["requested_class"],
                 tier="unknown" if unknown else "silver")
    if original["cohort"] == "natural":
        value.update(exact_votes=3, full_wake_veto=False)
    return value


class BoundaryTests(unittest.TestCase):
    def test_amendment_changes_only_routing_constants(self):
        original = (HERE / "english_v6_b21_continuation_primary_eval.py").read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(), "79a2a4e9cf28f7da40b985cadfb2e1c06e7fa1c50e52b4e91242f2abc48f7245")
        def bodies(raw):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        old, new = bodies(original), bodies(Path(primary.__file__).read_bytes())
        self.assertEqual(len(old), 35)  # 34 functions, including remote_file, plus Ledger.
        self.assertEqual(old, new)
        def assignments(raw):
            return {node.targets[0].id: ast.dump(node.value, include_attributes=False)
                    for node in ast.parse(raw).body
                    if isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)}
        old, new = assignments(original), assignments(Path(primary.__file__).read_bytes())
        self.assertEqual(old.keys(), new.keys())
        self.assertEqual({key for key in old if old[key] != new[key]}, {
            "SPEC", "PRODUCER", "WORKFLOW_TEMPLATE", "SOURCE_MODULE", "SOURCE_SCHEMA",
            "SCREEN_SCHEMA", "SOURCE_BRANCH", "SOURCE_WORKFLOW", "PRODUCER_ROOTS"})
        self.assertEqual(primary.SOURCE_MODULE, "english_v6_b21_natural_amendment_source")
        self.assertEqual(primary.SOURCE_BRANCH, "research/en-v6-b21-natural-amendment-sources-once-20261124")
        self.assertEqual(primary.BRANCH, "refs/heads/research/en-v6-b21-primary-scoring-once-20261124")
        self.assertEqual(primary.WORKFLOW, ".github/workflows/english-v6-b21-primary-model-eval-once.yml")
        self.assertEqual(primary.MARKER, "english-v6-b21-primary-consumed")
        self.assertEqual(primary.MAX_FORWARDS, 1440)

    def test_registration_stays_inert_or_has_complete_pins_and_both_failure_identities(self):
        spec = json.loads(primary.SPEC.read_bytes())
        producer = json.loads(primary.PRODUCER.read_bytes())
        self.assertIsNone(spec["prepared_parent"])
        self.assertTrue(all(spec[key] is False for key in primary.FLAGS))
        if spec["scoring_allowed"] is False:
            self.assertIsNone(spec["producer_manifest_sha256"])
            self.assertTrue(all(spec["inputs"][key] is None for key in ("pair", "source", "screen")))
            self.assertTrue(all(pin is None for pin in spec["native_inputs"].values()))
            self.assertTrue(all(spec["model_pt_sha256"][key] is None for key in ("control", "challenger")))
            self.assertFalse(producer["frozen"])
        else:
            self.assertEqual(primary.protocol(), spec)
            self.assertTrue(producer["frozen"])
            self.assertEqual(primary.sha(primary.PRODUCER.read_bytes()), spec["producer_manifest_sha256"])
        self.assertEqual(producer["source"]["workflow_path"], primary.SOURCE_WORKFLOW)
        origin = spec["transport_continuation"]
        self.assertEqual(origin["original_failed_source_run_id"], 45352507569)
        self.assertEqual(origin["failed_continuation_run_id"], 45365292922)
        self.assertEqual(origin["failed_continuation_head"], "2da5cbe0d616f96b21a1bb5665613b9ce8d1c788")
        self.assertTrue(origin["failed_continuation_preserved"])
        self.assertFalse(origin["failed_continuation_rerun_allowed"])

    def test_original_proof_zip_larger_than_contents_limit_uses_git_blob(self):
        raw = b"original-failure-proof" * 65000
        blob = primary.git_blob(raw)
        tree = dict(tree=[dict(path="research/original-producer.zip", type="blob", mode="100644", sha=blob)])
        response = dict(sha=blob, encoding="base64", size=len(raw), content=base64.b64encode(raw).decode())
        with patch.object(primary, "gh", return_value=json.dumps(response).encode()) as fetch:
            self.assertEqual(primary.remote_file("research/original-producer.zip", "a" * 40, primary.sha(raw), tree), raw)
            fetch.assert_called_once_with("git/blobs/" + blob)
        response["size"] -= 1
        with patch.object(primary, "gh", return_value=json.dumps(response).encode()):
            with self.assertRaises(ValueError): primary.remote_file("research/original-producer.zip", "a" * 40, primary.sha(raw), tree)

    def test_inert_is_before_dependencies_or_any_remote_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = Path(temporary) / "inert.json"
            spec.write_text('{"scoring_allowed":false}')
            with patch.object(primary, "SPEC", spec), patch.object(primary, "helpers", side_effect=AssertionError("helper")), \
                 patch.object(primary, "gh", side_effect=AssertionError("network")):
                with self.assertRaisesRegex(ValueError, "INERT"):
                    primary.fetch(Path(temporary) / "inputs", "a" * 40)
                with self.assertRaisesRegex(ValueError, "INERT"):
                    primary.preflight(Path(temporary) / "inputs", "a" * 40)
                with self.assertRaisesRegex(ValueError, "INERT"):
                    primary.score(Path(temporary) / "inputs", Path(temporary) / "result", "a" * 40)
                self.assertFalse((Path(temporary) / "inputs").exists())
                self.assertFalse((Path(temporary) / "result").exists())

    def test_source_join_preserves_unknown_and_natural_identity(self):
        originals = [row(1, positive=True), row(2), row(3, "natural")]
        decisions = [label(originals[0]), label(originals[1], unknown=True), label(originals[2])]
        selected = primary.exact_admitted_join(originals, decisions)
        self.assertEqual([value["id"] for value in selected], ["row-1", "row-3"])
        self.assertEqual(selected[1]["cell_id"], "natural-speech")
        changed = copy.deepcopy(decisions)
        changed[0]["source_group_id"] = "substituted"
        with self.assertRaises(ValueError):
            primary.exact_admitted_join(originals, changed)
        with self.assertRaises(ValueError):
            primary.exact_admitted_join(originals, list(reversed(decisions)))

    def test_natural_veto_and_boolean_class_are_rejected(self):
        natural = row(5, "natural")
        actual = label(natural)
        actual["full_wake_veto"] = True
        with self.assertRaises(ValueError):
            primary.exact_admitted_join([natural], [actual])
        ordinary = row(6)
        actual = label(ordinary)
        actual["observed_class"] = False
        with self.assertRaises(ValueError):
            primary.exact_admitted_join([ordinary], [actual])

    def test_full_source_rejects_are_in_overlap_denominator(self):
        originals = [dict(row(i + 100), source_speaker_id="p" + str(2000 + i // 4)) for i in range(96)]
        originals += [dict(row(i + 500, "natural"), speaker=str(3000 + i // 2)) for i in range(24)]
        training = [row(i + 1000) for i in range(88)]
        history = dict(all_pcm_sha256=[], all_source_groups=[], canonical_people=[])
        primary.reject_overlap(originals, training, history, {"99999"})
        leaked = copy.deepcopy(history)
        leaked["all_pcm_sha256"] = [originals[-1]["pcm_sha256"]]
        with self.assertRaises(ValueError):
            primary.reject_overlap(originals, training, leaked, {"99999"})
        with self.assertRaises(ValueError):
            primary.reject_overlap(originals, training, history, {originals[-1]["speaker"]})

    def test_consumption_precedes_forward_and_cannot_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = primary.Ledger(Path(temporary))
            ledger.current = dict(key="row1__1.0__control", role="control", source_id="row1", gain="1.0")
            ledger.entered("control")
            saved = json.loads((Path(temporary) / "row1__1.0__control.json").read_text())
            self.assertEqual(saved["status"], "consumed_unknown")
            self.assertEqual(ledger.counts["control"], 1)
            with self.assertRaises(ValueError):
                ledger.entered("control")
            ledger.saved(ledger.current["key"])
            with self.assertRaises(ValueError):
                ledger.entered("control")

    def test1440_hard_limit_is_before_new_consumption(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = primary.Ledger(Path(temporary))
            ledger.counts = dict.fromkeys(primary.ARMS, 360)
            ledger.current = dict(key="forbidden", role="B0", source_id="row1", gain="1.0")
            with self.assertRaisesRegex(ValueError, "hard ceiling"):
                ledger.entered("B0")
            self.assertEqual(sum(ledger.counts.values()), 1440)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_original_raw_archive_safety(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / "unsafe.zip"
            data = io.BytesIO()
            with zipfile.ZipFile(data, "w") as archive:
                archive.writestr("../outside", b"data")
            p.write_bytes(data.getvalue())
            with self.assertRaises(ValueError):
                primary.original_zip(p, dict(bytes=p.stat().st_size, sha256=primary.sha(p.read_bytes())))

    def test_ctc_event_clocks_keep_original_greedy_rule(self):
        result = primary.ctc_events([0, 1, 1, 0, 2, 0, 3, 3, 0])
        self.assertEqual(result, dict(tokens=[1, 2, 3], event_frames=[6], event_samples=[1472], events=1))

    def test_first_public_single_parent_event_required(self):
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=primary.REPO, GITHUB_EVENT_NAME="push",
                   GITHUB_REF=primary.BRANCH, GITHUB_RUN_ATTEMPT="1", GITHUB_SHA="b" * 40, GITHUB_RUN_ID="123")
        event = dict(before="a" * 40, after="b" * 40, ref=primary.BRANCH, forced=False, deleted=False,
                     repository=dict(full_name=primary.REPO, private=False))
        spec = dict(prepared_parent=None, parent_authority="reviewed_workflow")
        identity = primary.guard(spec, "a" * 40, env, event, "b" * 40, "a" * 40)
        self.assertEqual(identity["run_id"], "123")
        for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_EVENT_NAME", "workflow_dispatch")):
            with self.assertRaises(ValueError):
                primary.guard(spec, "a" * 40, dict(env, **{key: value}), event, "b" * 40, "a" * 40)

    def test_real_git_blob_required_for_hidden_producer_file(self):
        raw = b"hidden source metadata\n"
        tree = dict(tree=[dict(path=".github/triggers/a.json", type="blob", mode="100644", sha=primary.git_blob(raw))])
        self.assertEqual(primary.tree_member(tree, ".github/triggers/a.json"), primary.git_blob(raw))
        with self.assertRaises(ValueError):
            primary.tree_member(tree, "missing")

    def test_workflow_retains_all_hidden_original_producer_files(self):
        template = primary.WORKFLOW_TEMPLATE.read_text()
        self.assertEqual(template.count(primary.PARENT_TOKEN), 2)
        self.assertIn("include-hidden-files: true", template)
        self.assertIn("primary_producer/", template)
        self.assertIn("source_producer/", template)
        self.assertIn("english-v6-b21-primary-consumed", template)
        self.assertNotIn("workflow_dispatch", template)


if __name__ == "__main__":
    unittest.main()
