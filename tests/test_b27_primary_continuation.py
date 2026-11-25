"""Invented metadata/bytes only; no network, model or original scientific reader."""
from __future__ import annotations

import base64
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import b27_primary_continuation as helper

p = helper.profile


def config():
    return dict(schema=helper.CONFIG_SCHEMA, enabled=True, primary_run_id=123,
        primary_run_head="1" * 40, primary_prepared_parent="2" * 40,
        **{key: "a" * 64 for key in helper.HASH_FILES})


def pin(ident, run=123, head="1" * 40):
    return dict(id=ident, run_id=run, run_head=head, bytes=1, sha256=p.sha(b"z"))


def api_fixture():
    cfg = config()
    run = dict(id=123, head_sha="1" * 40, run_attempt=1, event="push",
        status="completed", conclusion="success", path=helper.evaluator.WORKFLOW,
        head_branch=helper.evaluator.BRANCH.removeprefix("refs/heads/"),
        repository=dict(full_name=helper.REPO, private=False))
    commit = dict(sha="1" * 40, parents=[dict(sha="2" * 40)],
                  files=[dict(filename=helper.evaluator.WORKFLOW)])
    items = [dict(id=100 + i, name=name, size_in_bytes=1, expired=False,
                  digest="sha256:" + p.sha(b"z"),
                  workflow_run=dict(id=123, head_sha="1" * 40))
             for i, name in enumerate(helper.quality.artifact_names("primary", 123).values())]
    return cfg, run, commit, dict(total_count=3, artifacts=items)


def branch_fixture():
    cfg = config()
    members = {key: (key + "\n").encode() for key in helper.HASH_FILES}
    cfg.update({key: p.sha(raw) for key, raw in members.items()})
    outputs = {role: pin(100 + i) for i, role in enumerate(helper.OUTPUT_ROLES)}
    lock = helper.build_lock(cfg, outputs, members, b'{"fixture":"result"}')
    train_raw = p.canonical(dict(fixture="saved training report"))
    bound = dict(
        assessment_sha256="e" * 64,
        training_lock=dict(training=dict(model_pt_sha256={
            "B0": "a" * 64, "B9": "b" * 64, "control": "c" * 64, "challenger": "d" * 64})),
        files={"fixture.zip": dict(bytes=1, sha256=p.sha(b"z"))},
        transport={"fixture.zip": dict(artifact_id=999)},
        training_report_sha256=p.sha(train_raw))
    api = {key: b"{}" for key in ("run", "commit", "artifacts")}
    inputs = {role: pin(200 + i) for i, role in enumerate(sorted(helper.INPUT_ROLES))}
    files, transport = helper.quality_layout(api, outputs, inputs)
    return cfg, lock, bound, train_raw, files, transport


def report_fixture(gate, lock, bound):
    report = dict(
        schema=helper.quality.REPORT_SCHEMA, panel="primary", panel_gate=gate,
        original_lock_sha256=p.sha(p.canonical(lock)),
        original_result_sha256=lock["original_result_sha256"],
        external_proof_binding_sha256=lock["external_proof_binding_sha256"],
        original_training_report_sha256=bound["training_report_sha256"],
        model_pt_sha256=bound["training_lock"]["training"]["model_pt_sha256"],
        assessment_sha256=bound["assessment_sha256"],
        replication_open_allowed=gate == "PASS",
        candidate_engineering_validation_allowed=False, automatic_next_assessment_allowed=False,
        promotion_allowed=False, product_released=False,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0)
    report.update(dict.fromkeys((
        "all_original_arrays_verified", "original_source_admission_recomputed",
        "original_producer_git_verified", "all_original_training_proof_verified",
        "complete_original_Q1_failure_verified", "old_stops_and_reserves_preserved"), True))
    return report


class EntryAndMetadata(unittest.TestCase):
    def test_inert_config_stops_before_network_or_directory_creation(self):
        inert = dict(schema=helper.CONFIG_SCHEMA, enabled=False,
                     **dict.fromkeys(helper.CONFIG_PINS))
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "never-created"
            with mock.patch.object(helper.evaluator, "gh") as network, \
                 mock.patch.object(helper.original_proof, "fetch_files") as fetch, \
                 mock.patch.object(helper, "certify_once") as certify, \
                 mock.patch.object(Path, "mkdir") as mkdir:
                with self.assertRaisesRegex(ValueError, "INERT"):
                    helper.prepare(inert, work)
            network.assert_not_called()
            fetch.assert_not_called()
            certify.assert_not_called()
            mkdir.assert_not_called()
            self.assertFalse(work.exists())

    def test_configuration_is_exact_and_has_no_runtime_override(self):
        repository_config = p.document(helper.CONFIG.read_bytes())
        if repository_config["enabled"] is False:
            self.assertTrue(all(repository_config[key] is None for key in helper.CONFIG_PINS))
            with self.assertRaisesRegex(ValueError, "INERT"):
                helper.validate_config(repository_config)
        else:
            helper.validate_config(repository_config)
        self.assertEqual(helper.validate_config(config()), config())
        for key, value in (
            ("enabled", 1), ("primary_run_id", True), ("primary_run_id", 0),
            ("primary_run_head", "g" * 40), ("inputs_sha256", None),
            ("primary_prepared_parent", "1" * 40)):
            bad = dict(config(), **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                helper.validate_config(bad)
        with self.assertRaises(ValueError):
            helper.validate_config(dict(config(), threshold=0.0))

    def test_separate_workflow_only_first_push_guard(self):
        head, parent = "3" * 40, "4" * 40
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=helper.REPO,
            GITHUB_EVENT_NAME="push", GITHUB_REF=helper.BRANCH, GITHUB_RUN_ATTEMPT="1",
            GITHUB_SHA=head, PREPARED_PARENT=parent)
        event = dict(before=parent, after=head, ref=helper.BRANCH,
            forced=False, deleted=False, repository=dict(full_name=helper.REPO, private=False))
        helper.validate_entry(env, event, head, parent, [helper.WORKFLOW])
        for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_REF", helper.evaluator.BRANCH),
                           ("GITHUB_EVENT_NAME", "pull_request"), ("PREPARED_PARENT", "5" * 40)):
            with self.assertRaises(ValueError):
                helper.validate_entry(dict(env, **{key: value}), event, head, parent, [helper.WORKFLOW])
        for changed in ([], [helper.WORKFLOW, "automation/changed.py"]):
            with self.assertRaises(ValueError):
                helper.validate_entry(env, event, head, parent, changed)
        for flag in ("forced", "deleted"):
            with self.assertRaises(ValueError):
                helper.validate_entry(env, dict(event, **{flag: True}), head, parent, [helper.WORKFLOW])

    def test_activation_changes_only_two_parent_tokens(self):
        parent = "4" * 40
        before = ("before=" + helper.PARENT_TOKEN + "\nparent=" +
                  helper.PARENT_TOKEN + "\n").encode("ascii")
        current = before.replace(helper.PARENT_TOKEN.encode("ascii"), parent.encode("ascii"))
        helper.validate_workflow_activation(before, current, parent)
        for old, new in ((before, current + b"extra\n"), (before, before),
                         (before + helper.PARENT_TOKEN.encode("ascii"), current)):
            with self.assertRaises(ValueError):
                helper.validate_workflow_activation(old, new, parent)

    def test_final_run_rejects_unknown_failed_replayed_and_substituted_identity(self):
        cfg, run, commit, inventory = api_fixture()
        pins = helper.validate_primary_api(cfg, run, commit, inventory)
        self.assertEqual(set(pins), set(helper.OUTPUT_ROLES))
        for key, value in (
            ("status", "in_progress"), ("conclusion", None), ("conclusion", "failure"),
            ("run_attempt", 2), ("id", 124), ("head_sha", "0" * 40),
            ("event", "workflow_dispatch"), ("path", helper.WORKFLOW),
            ("head_branch", helper.BRANCH.removeprefix("refs/heads/"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                helper.validate_primary_api(cfg, dict(run, **{key: value}), commit, inventory)
        bad = copy.deepcopy(run)
        bad["repository"]["private"] = True
        with self.assertRaises(ValueError):
            helper.validate_primary_api(cfg, bad, commit, inventory)
        bad_commit = copy.deepcopy(commit)
        bad_commit["files"].append(dict(filename="automation/changed.py"))
        with self.assertRaises(ValueError):
            helper.validate_primary_api(cfg, run, bad_commit, inventory)

    def test_three_artifact_inventory_is_complete_unique_and_authentic(self):
        cfg, run, commit, original = api_fixture()
        cases = []
        bad = copy.deepcopy(original); bad["artifacts"].pop(); cases.append(bad)
        bad = copy.deepcopy(original); bad["artifacts"].append(dict(bad["artifacts"][0])); cases.append(bad)
        for field, value in (("expired", True), ("digest", "sha256:bad"),
                             ("size_in_bytes", 0), ("name", "substitute")):
            bad = copy.deepcopy(original); bad["artifacts"][0][field] = value; cases.append(bad)
        bad = copy.deepcopy(original)
        bad["artifacts"][1]["id"] = bad["artifacts"][0]["id"]; cases.append(bad)
        bad = copy.deepcopy(original)
        bad["artifacts"][0]["workflow_run"]["head_sha"] = "9" * 40; cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):
                helper.validate_primary_api(cfg, run, commit, bad)

    def test_lock_hashes_original_member_bytes_without_reserializing(self):
        cfg = config()
        members = {key: (key + "\n").encode() for key in helper.HASH_FILES}
        cfg.update({key: p.sha(raw) for key, raw in members.items()})
        outputs = {role: pin(100 + i) for i, role in enumerate(helper.OUTPUT_ROLES)}
        raw = b'{ "panel_gate" : "PASS" }\n'
        lock = helper.build_lock(cfg, outputs, members, raw)
        self.assertEqual(set(lock), helper.quality.LOCK_FIELDS)
        self.assertEqual(len(lock), 10)
        self.assertEqual(lock["original_result_sha256"], p.sha(raw))
        self.assertNotEqual(lock["original_result_sha256"], p.sha(p.canonical(p.document(raw))))
        for key in members:
            with self.subTest(key=key), self.assertRaises(ValueError):
                helper.build_lock(cfg, outputs, {**members, key: members[key] + b"x"}, raw)

    def test_raw_gate_is_routing_only_and_rejects_partial_or_wrong_candidate(self):
        cfg, lock, bound, _, _, _ = branch_fixture()
        spec = dict(schema=helper.evaluator.SCHEMA, panel="primary", scoring_allowed=True,
            producer_manifest_sha256=lock["producer_manifest_sha256"],
            external_proof_binding_sha256=lock["external_proof_binding_sha256"],
            model_pt_sha256=bound["training_lock"]["training"]["model_pt_sha256"],
            assessment_sha256=bound["assessment_sha256"])
        result = dict(schema=helper.evaluator.SCHEMA, run_id=str(cfg["primary_run_id"]),
            run_head=cfg["primary_run_head"], run_attempt=1,
            prepared_parent=cfg["primary_prepared_parent"],
            registered_inputs_sha256=lock["inputs_sha256"],
            producer_manifest_sha256=lock["producer_manifest_sha256"],
            implementation_sha256=lock["evaluator_sha256"],
            model_pt_sha256=spec["model_pt_sha256"], assessment_sha256=spec["assessment_sha256"],
            source_gate="SOURCE_GATE_PASS", source_shortfall={}, panel_gate="PASS")
        for gate in ("PASS", "NO_PROMOTION"):
            self.assertEqual(helper.preliminary_gate(cfg, dict(result, panel_gate=gate), spec, lock, bound), gate)
        for gate in ("STOP", None, "UNKNOWN"):
            with self.assertRaises(ValueError):
                helper.preliminary_gate(cfg, dict(result, panel_gate=gate), spec, lock, bound)
        with self.assertRaises(ValueError):
            helper.preliminary_gate(cfg, dict(result, model_pt_sha256={}), spec, lock, bound)


class OriginalClosure(unittest.TestCase):
    def test_twelve_input_pins_preserve_original_baselines_and_train(self):
        _, _, bound, _, _, _ = branch_fixture()
        pins = {role: pin(1000 + i, run=55, head="6" * 40)
                for i, role in enumerate(sorted(helper.INPUT_ROLES))}
        pins.update(helper.evaluator.BASELINES)
        training = bound["training_lock"]["training"]
        training.update(run_id=55, run_head="6" * 40,
            artifacts={role: {key: pins[alias][key] for key in ("id", "bytes", "sha256")}
                       for alias, role in (("pair", "models"), ("train_producer", "producer"))})
        spec = dict(inputs={key: pins[key] for key in ("B0", "B9", "pair", "source", "screen")},
            native_inputs={key: pins["native_" + key] for key in helper.PROVIDERS},
            upstream_producers={key: pins[key + "_producer"] for key in ("train", "source")})
        self.assertEqual(helper.input_pins(spec, bound), pins)
        bad = copy.deepcopy(spec); bad["native_inputs"].pop("qwen")
        with self.assertRaises(ValueError):
            helper.input_pins(bad, bound)
        bad = copy.deepcopy(spec); bad["inputs"]["B0"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            helper.input_pins(bad, bound)
        bad = copy.deepcopy(spec)
        bad["inputs"]["source"]["bytes"] = helper.quality.archive_limit("primary", "source") + 1
        with self.assertRaises(ValueError):
            helper.input_pins(bad, bound)

    def test_exact_eighteen_file_layout_raw_api_bytes_and_no_orphan(self):
        api = dict(run=b'{ "r": 1 }\n', commit=b"{}", artifacts=b'{"items":[]}')
        outputs = {role: pin(100 + i) for i, role in enumerate(helper.OUTPUT_ROLES)}
        inputs = {role: pin(200 + i) for i, role in enumerate(sorted(helper.INPUT_ROLES))}
        files, transport = helper.quality_layout(api, outputs, inputs)
        self.assertEqual(len(files), 18)
        self.assertEqual(base64.b64decode(transport["api/run.json"]["original_api_base64"]), api["run"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in files:
                target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(api[Path(name).stem] if name.startswith("api/") else b"z")
            helper.require_quality_inventory(root, files)
            extra = root / "PRIMARY_LOCK.json"; extra.write_bytes(b"{}")
            with self.assertRaises(ValueError):
                helper.require_quality_inventory(root, files)
            extra.unlink()
            missing = root / "inputs/native_qwen.zip"; missing.unlink()
            with self.assertRaises(ValueError):
                helper.require_quality_inventory(root, files)
            missing.symlink_to(root / "inputs/source.zip")
            with self.assertRaises(ValueError):
                helper.require_quality_inventory(root, files)
        with self.assertRaises(ValueError):
            helper.quality_layout(api, outputs, {k: v for k, v in inputs.items() if k != "source"})
        bad = copy.deepcopy(inputs); bad["source"]["id"] = outputs["result"]["id"]
        with self.assertRaises(ValueError):
            helper.quality_layout(api, outputs, bad)


class CertificationRoutes(unittest.TestCase):
    def test_no_promotion_calls_quality_once_and_never_replica_binder(self):
        _, lock, bound, _, files, transport = branch_fixture()
        report = report_fixture("NO_PROMOTION", lock, bound)
        with tempfile.TemporaryDirectory() as temporary:
            proof, output = Path(temporary) / "proof", Path(temporary) / "bound"
            with mock.patch.object(helper.quality, "verify_quality_original", return_value=report) as verify, \
                 mock.patch.object(helper.source_binder, "bind", side_effect=AssertionError("must stay sealed")) as bind:
                raw, extra = helper.certify_once("NO_PROMOTION", lock, proof, output, bound, files, transport)
            verify.assert_called_once_with(proof / "primary-quality", lock, proof)
            bind.assert_not_called()
            self.assertEqual(p.document(raw), report)
            self.assertEqual(extra, {})
            self.assertFalse(output.exists())

    def test_pass_calls_frozen_binder_once_without_preliminary_quality_reader(self):
        _, lock, bound, train_raw, files, transport = branch_fixture()
        report = report_fixture("PASS", lock, bound)

        def fake_bind(panel, proof, training_lock, output, primary_lock=None):
            self.assertEqual(panel, "replication")
            self.assertEqual(training_lock, bound["training_lock"])
            self.assertEqual(primary_lock, lock)
            replica = dict(panel="replication", status="BOUND_COMPLETE_ORIGINAL_TRAIN",
                scientific_execution_authorized=False, training_lock=training_lock,
                files=bound["files"], transport=bound["transport"],
                training_report_sha256=bound["training_report_sha256"],
                primary_quality_lock=lock, primary_quality_files=files,
                primary_quality_transport=transport,
                primary_quality_report_sha256=p.sha(p.canonical(report)))
            binding_raw = p.canonical(replica)
            audit = dict(binding_sha256=p.sha(binding_raw), original_primary_pass_required=True,
                scientific_execution_authorized=False, new_model_calls=0, new_source_calls=0,
                new_observer_calls=0, remote_mutations=0)
            payloads = {helper.REPLICA_PATH: binding_raw,
                "PRIMARY_READBACK.json": p.canonical(report),
                "TRAINING_READBACK.json": train_raw, "BINDING_AUDIT.json": p.canonical(audit)}
            output.mkdir()
            for name, raw in payloads.items():
                target = output / name; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
            return audit

        with tempfile.TemporaryDirectory() as temporary:
            proof, output = Path(temporary) / "proof", Path(temporary) / "bound"
            with mock.patch.object(helper.source_binder, "bind", side_effect=fake_bind) as bind, \
                 mock.patch.object(helper.quality, "verify_quality_original",
                                   side_effect=AssertionError("no additional PRIMARY readback")) as verify:
                raw, extra = helper.certify_once("PASS", lock, proof, output, bound, files, transport)
            bind.assert_called_once_with("replication", proof, bound["training_lock"], output, primary_lock=lock)
            verify.assert_not_called()
            self.assertEqual(p.document(raw), report)
            self.assertEqual(set(extra), set(helper.PASS_PATHS))

    def test_failed_certification_never_falls_back_or_opens_another_route(self):
        _, lock, bound, _, files, transport = branch_fixture()
        with mock.patch.object(helper.source_binder, "bind", side_effect=ValueError("bad original")) as bind, \
             mock.patch.object(helper.quality, "verify_quality_original") as verify:
            with self.assertRaisesRegex(ValueError, "bad original"):
                helper.certify_once("PASS", lock, Path("/fixture"), Path("/unused"), bound, files, transport)
            bind.assert_called_once()
            verify.assert_not_called()
        wrong = report_fixture("PASS", lock, bound)
        with mock.patch.object(helper.quality, "verify_quality_original", return_value=wrong) as verify, \
             mock.patch.object(helper.source_binder, "bind") as bind:
            with self.assertRaises(ValueError):
                helper.certify_once("NO_PROMOTION", lock, Path("/fixture"), Path("/unused"), bound, files, transport)
            verify.assert_called_once()
            bind.assert_not_called()
        with mock.patch.object(helper.quality, "verify_quality_original") as verify, \
             mock.patch.object(helper.source_binder, "bind") as bind:
            with self.assertRaises(ValueError):
                helper.certify_once("STOP", lock, Path("/fixture"), Path("/unused"), bound, files, transport)
            verify.assert_not_called()
            bind.assert_not_called()

    def test_verified_report_requires_same_lock_hashes_flags_and_zero_new_calls(self):
        _, lock, bound, _, _, _ = branch_fixture()
        original = report_fixture("NO_PROMOTION", lock, bound)
        for key, value in (
            ("original_lock_sha256", "0" * 64), ("original_result_sha256", "0" * 64),
            ("original_training_report_sha256", "0" * 64), ("all_original_arrays_verified", False),
            ("replication_open_allowed", True), ("promotion_allowed", True), ("new_model_calls", 1),
            ("new_source_calls", False), ("model_pt_sha256", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                helper.check_report(dict(original, **{key: value}), "NO_PROMOTION", lock, bound)


class ExportProtocol(unittest.TestCase):
    def test_five_file_terminal_and_large_eight_file_pass_roundtrip(self):
        for outcome in ("NO_PROMOTION", "PASS"):
            paths = helper.COMMON_PATHS + (helper.PASS_PATHS if outcome == "PASS" else ())
            files = {path: b'{"invented":true}\n' for path in paths}
            if outcome == "PASS":
                files[helper.REPLICA_PATH] = b"x" * 2_300_000
            records, manifest = helper.make_export(files, outcome)
            self.assertEqual(manifest["outcome"], outcome)
            self.assertEqual(manifest["file_count"], len(paths))
            self.assertEqual(manifest["total_bytes"], sum(map(len, files.values())))
            for item in manifest["files"]:
                chunks = [r for r in records if r["path"] == item["path"]]
                self.assertEqual([r["index"] for r in chunks], list(range(item["chunks"])))
                self.assertTrue(all(len(r["base64"]) <= helper.CHUNK_CHARS for r in chunks))
                self.assertTrue(all({key: row[key] for key in item} == item for row in chunks))
                raw = base64.b64decode("".join(r["base64"] for r in chunks), validate=True)
                self.assertEqual(raw, files[item["path"]])
                self.assertEqual(len(raw), item["bytes"])
                self.assertEqual(p.sha(raw), item["sha256"])
            if outcome == "NO_PROMOTION":
                self.assertFalse(set(helper.PASS_PATHS) & set(files))

    def test_wrong_outcome_whitelist_empty_and_oversize_exports_are_rejected(self):
        files = {path: b"x" for path in helper.COMMON_PATHS}
        cases = [
            ("STOP", files),
            ("PASS", files),
            ("NO_PROMOTION", {**files, helper.REPLICA_PATH: b"x"}),
            ("NO_PROMOTION", {key: value for key, value in files.items() if key != helper.LOCK_PATH}),
            ("NO_PROMOTION", {**files, helper.REPORT_PATH: b""}),
            ("NO_PROMOTION", {**files, helper.RESULT_PATH: b"x" * helper.MAX_EXPORT_BYTES}),
        ]
        for outcome, payload in cases:
            with self.assertRaises(ValueError):
                helper.make_export(payload, outcome)


if __name__ == "__main__":
    unittest.main()
