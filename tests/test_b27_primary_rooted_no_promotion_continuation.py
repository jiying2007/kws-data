"""Invented bytes/metadata only: no network, adapter reader or scientific calls."""
from __future__ import annotations

import base64
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import b27_primary_rooted_no_promotion_continuation as helper


def config():
    return dict(schema=helper.CONFIG_SCHEMA, enabled=True, **helper.ORIGINAL,
                rooted_adapter_sha256="a" * 64, rooted_adapter_bytes=100)


def inert():
    return dict(schema=helper.CONFIG_SCHEMA, enabled=False, **dict.fromkeys(helper.CONFIG_PINS))


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = helper.ROOT / helper.HELPER_PATH
        raw = helper.verified_bytes(path, helper.HELPER_SHA)
        cls.original = helper.load_bytes(path, raw, "_b27_frozen_transport_contract_fixture")

    def runtime(self):
        adapter = types.SimpleNamespace(verify_original=mock.Mock())
        return helper.bind_private(self.original, adapter, config()), adapter

    def fixture(self):
        ns, adapter = self.runtime()
        p = ns["profile"]
        lock = dict(schema=ns["quality"].LOCK_SCHEMA, panel="primary",
            **{key: str(i) * 64 for i, key in enumerate(ns["HASH_FILES"], 1)},
            original_result_sha256=p.sha(b'{"invented":"result"}'))
        for i, role in enumerate(ns["OUTPUT_ROLES"], 100):
            lock[role] = dict(id=i, bytes=1, sha256=p.sha(b"z"), run_id=123, run_head="1" * 40)
        bound = dict(training_report_sha256="e" * 64, assessment_sha256="f" * 64,
            training_lock=dict(training=dict(model_pt_sha256={name: str(i) * 64
                for i, name in enumerate(("B0", "B9", "control", "challenger"), 1)})))
        report = dict(schema=ns["quality"].REPORT_SCHEMA, panel="primary", panel_gate="NO_PROMOTION",
            original_lock_sha256=p.sha(p.canonical(lock)), original_result_sha256=lock["original_result_sha256"],
            external_proof_binding_sha256=lock["external_proof_binding_sha256"],
            original_training_report_sha256=bound["training_report_sha256"],
            model_pt_sha256=bound["training_lock"]["training"]["model_pt_sha256"],
            assessment_sha256=bound["assessment_sha256"], actual_forwards=996, verified_frame_count=178776,
            **dict.fromkeys(("all_original_arrays_verified", "original_source_admission_recomputed",
                "original_producer_git_verified", "all_original_training_proof_verified",
                "complete_original_Q1_failure_verified", "old_stops_and_reserves_preserved"), True),
            **dict.fromkeys(("replication_open_allowed", "candidate_engineering_validation_allowed",
                "automatic_next_assessment_allowed", "promotion_allowed", "product_released"), False),
            **dict.fromkeys(("new_model_calls", "new_source_calls", "new_observer_calls"), 0))
        rooted = dict(schema=helper.ROOTED_SCHEMA, panel="primary", panel_gate="NO_PROMOTION",
            original_reader_sha256=helper.READER_SHA, import_scope_adapter_sha256=config()["rooted_adapter_sha256"],
            original_report=report, original_report_sha256=p.sha(p.canonical(report)),
            original_failed_continuation_run_id=helper.FAILED_CONTINUATION["run_id"],
            **dict.fromkeys(helper.CLOSED_FLAGS, False), **dict.fromkeys(helper.ZERO_COUNTERS, 0))
        adapter.verify_original.return_value = rooted
        return ns, adapter, lock, bound, report, rooted

    def exports(self):
        ns, adapter, lock, bound, report, rooted = self.fixture()
        p = ns["profile"]
        receipt = dict(primary_outcome="NO_PROMOTION", candidate_terminal=True,
            replica_proof_ready_for_review=False, certification_route="frozen_primary_quality_reader",
            scientific_execution_authorized=False, replication_source_started=False, engineering_started=False,
            **dict.fromkeys(helper.ZERO_COUNTERS, 0), remote_mutations=0,
            promotion_allowed=False, product_released=False, automatic_next_assessment_allowed=False)
        files = {ns["LOCK_PATH"]: p.canonical(lock), ns["REPORT_PATH"]: p.canonical(report),
            ns["RESULT_PATH"]: b'{"invented":"result"}', helper.ROOTED_PATH: p.canonical(rooted),
            ns["RECEIPT_PATH"]: p.canonical(receipt),
            ns["INVENTORY_PATH"]: p.canonical(dict(training_origin_files={"invented": "unchanged"},
                primary_quality_files={"invented": "unchanged"}, review_files={}))}
        return ns, adapter, files

    def test_disabled_entries_reject_before_runtime_or_workspace_access(self):
        with mock.patch.object(helper, "runtime", side_effect=AssertionError("no runtime before binding")) as runtime, \
             mock.patch.object(Path, "mkdir", side_effect=AssertionError("no directory")) as mkdir:
            with self.assertRaisesRegex(ValueError, "INERT"):
                helper.prepare(inert(), object())
            with mock.patch.object(helper, "CONFIG", types.SimpleNamespace(
                    read_bytes=lambda: self.original.profile.canonical(inert()))):
                with self.assertRaisesRegex(ValueError, "INERT"):
                    helper.main()
            runtime.assert_not_called()
            mkdir.assert_not_called()
        with mock.patch.object(helper, "verified_bytes", side_effect=AssertionError("no implementation read")) as read, \
             mock.patch.object(helper, "load_bytes", side_effect=AssertionError("no implementation import")) as load:
            with self.assertRaisesRegex(ValueError, "INERT"):
                helper.runtime(inert())
            read.assert_not_called()
            load.assert_not_called()

    def test_exact_config_requires_one_original_identity_and_reviewed_adapter(self):
        actual = helper.document(helper.CONFIG.read_bytes())
        if actual["enabled"]:
            helper.validate_config(actual)
        else:
            self.assertTrue(all(actual[key] is None for key in helper.CONFIG_PINS))
            with self.assertRaisesRegex(ValueError, "INERT"):
                helper.validate_config(actual)
        self.assertEqual(helper.validate_config(config()), config())
        for key, value in (("enabled", 1), ("primary_run_id", 45643969797),
                ("primary_run_id", 45627775991.0), ("primary_run_head", "0" * 40),
                ("original_helper_sha256", "0" * 64), ("original_reader_sha256", "0" * 64),
                ("rooted_adapter_sha256", None), ("rooted_adapter_sha256", "g" * 64),
                ("rooted_adapter_bytes", True), ("rooted_adapter_bytes", 0)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                helper.validate_config(dict(config(), **{key: value}))
        for bad in (dict(config(), allow_pass=True), dict(inert(), rooted_adapter_sha256="a" * 64)):
            with self.assertRaises(ValueError):
                helper.validate_config(bad)

    def test_changed_implementation_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.py"
            raw = b'raise AssertionError("must never execute")\n'
            path.write_bytes(raw)
            with mock.patch.object(helper, "load_bytes", side_effect=AssertionError("no unverified code")) as execute:
                for digest, size in (("0" * 64, len(raw)), (helper.sha(raw), len(raw) + 1)):
                    with self.assertRaises(ValueError):
                        helper.verified_bytes(path, digest, size)
                execute.assert_not_called()
            self.assertEqual(helper.verified_bytes(path, helper.sha(raw), len(raw)), raw)
            link = path.with_name("symlink.py"); link.symlink_to(path)
            with self.assertRaises(ValueError):
                helper.verified_bytes(link, helper.sha(raw))

    def test_private_namespace_keeps_all_original_transport_code_and_globals(self):
        before = dict(vars(self.original))
        ns, _adapter = self.runtime()
        self.assertEqual(set(vars(self.original)), set(before))
        self.assertTrue(all(vars(self.original)[key] is value for key, value in before.items()))
        for name in ("prepare", "main", "validate_entry", "validate_workflow_activation", "validate_run",
                     "validate_primary_api", "build_lock", "input_pins", "quality_layout",
                     "require_quality_inventory", "check_report"):
            self.assertIs(ns[name].__code__, getattr(self.original, name).__code__)
            self.assertIs(ns[name].__globals__, ns)
            self.assertIsNot(ns[name].__globals__, vars(self.original))
        self.assertEqual(len(self.original.COMMON_PATHS), 5)
        self.assertEqual(len(self.original.PASS_PATHS), 3)
        self.assertEqual(ns["COMMON_PATHS"], (*self.original.COMMON_PATHS, helper.ROOTED_PATH))
        self.assertEqual(ns["PASS_PATHS"], ())
        self.assertNotEqual(self.original.BRANCH, ns["BRANCH"])

    def test_separate_first_push_and_exact_two_token_workflow_guard(self):
        ns, _adapter = self.runtime()
        head, parent = "3" * 40, "4" * 40
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=self.original.REPO,
            GITHUB_EVENT_NAME="push", GITHUB_REF=helper.BRANCH, GITHUB_RUN_ATTEMPT="1",
            GITHUB_SHA=head, PREPARED_PARENT=parent)
        event = dict(before=parent, after=head, ref=helper.BRANCH, forced=False, deleted=False,
                     repository=dict(full_name=self.original.REPO, private=False))
        ns["validate_entry"](env, event, head, parent, [helper.WORKFLOW])
        for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_REF", self.original.BRANCH),
                           ("GITHUB_EVENT_NAME", "workflow_dispatch"), ("PREPARED_PARENT", "5" * 40)):
            with self.assertRaises(ValueError):
                ns["validate_entry"](dict(env, **{key: value}), event, head, parent, [helper.WORKFLOW])
        for changed in ([], [helper.WORKFLOW, helper.ADAPTER_PATH]):
            with self.assertRaises(ValueError):
                ns["validate_entry"](env, event, head, parent, changed)
        for flag in ("forced", "deleted"):
            with self.assertRaises(ValueError):
                ns["validate_entry"](env, dict(event, **{flag: True}), head, parent, [helper.WORKFLOW])
        before = ("before=" + helper.PARENT_TOKEN + "\nparent=" + helper.PARENT_TOKEN + "\n").encode()
        after = before.replace(helper.PARENT_TOKEN.encode(), parent.encode())
        ns["validate_workflow_activation"](before, after, parent)
        for old, new in ((before, after + b"extra"), (before, before),
                         (before + helper.PARENT_TOKEN.encode(), after)):
            with self.assertRaises(ValueError):
                ns["validate_workflow_activation"](old, new, parent)

    def test_primary_api_stays_the_original_terminal_attempt_one(self):
        ns, _adapter = self.runtime()
        cfg = config()
        run = dict(id=cfg["primary_run_id"], head_sha=cfg["primary_run_head"], run_attempt=1,
            path=self.original.evaluator.WORKFLOW, event="push", status="completed", conclusion="success",
            head_branch=self.original.evaluator.BRANCH.removeprefix("refs/heads/"),
            repository=dict(full_name=self.original.REPO, private=False))
        ns["validate_run"](cfg, run)
        for key, value in (("id", 45643969797), ("run_attempt", 2), ("status", "in_progress"),
                           ("conclusion", "failure"), ("path", helper.WORKFLOW)):
            with self.assertRaises(ValueError):
                ns["validate_run"](cfg, dict(run, **{key: value}))

    def test_raw_pass_and_incomplete_routes_reject_before_adapter(self):
        ns, adapter, lock, bound, _report, _rooted = self.fixture()
        cfg = config()
        spec = dict(schema=self.original.evaluator.SCHEMA, panel="primary", scoring_allowed=True,
            producer_manifest_sha256=lock["producer_manifest_sha256"],
            external_proof_binding_sha256=lock["external_proof_binding_sha256"],
            model_pt_sha256=bound["training_lock"]["training"]["model_pt_sha256"],
            assessment_sha256=bound["assessment_sha256"])
        result = dict(schema=spec["schema"], run_id=str(cfg["primary_run_id"]), run_head=cfg["primary_run_head"],
            run_attempt=1, prepared_parent=cfg["primary_prepared_parent"], registered_inputs_sha256=lock["inputs_sha256"],
            producer_manifest_sha256=lock["producer_manifest_sha256"], implementation_sha256=lock["evaluator_sha256"],
            model_pt_sha256=spec["model_pt_sha256"], assessment_sha256=spec["assessment_sha256"],
            source_gate="SOURCE_GATE_PASS", source_shortfall={}, panel_gate="NO_PROMOTION")
        self.assertEqual(ns["preliminary_gate"](cfg, result, spec, lock, bound), "NO_PROMOTION")
        for gate in ("PASS", "STOP", None):
            with self.assertRaises(ValueError):
                ns["preliminary_gate"](cfg, dict(result, panel_gate=gate), spec, lock, bound)
            with self.assertRaises(ValueError):
                ns["certify_once"](gate, lock, Path("/invented"), Path("/unused"), bound, {}, {})
        adapter.verify_original.assert_not_called()

    def test_adapter_called_once_inner_and_outer_reports_retained_no_fallback(self):
        ns, adapter, lock, bound, report, rooted = self.fixture()
        direct = mock.Mock(side_effect=AssertionError("no standalone reader"))
        binder = mock.Mock(side_effect=AssertionError("no replica binder"))
        ns["quality"] = types.SimpleNamespace(REPORT_SCHEMA=self.original.quality.REPORT_SCHEMA,
                                               verify_quality_original=direct)
        ns["source_binder"] = types.SimpleNamespace(bind=binder)
        proof = Path("/invented-proof")
        raw, extra = ns["certify_once"]("NO_PROMOTION", lock, proof, Path("/unused"), bound, {}, {})
        adapter.verify_original.assert_called_once_with(proof / "primary-quality", lock, proof)
        direct.assert_not_called(); binder.assert_not_called()
        self.assertEqual(raw, ns["profile"].canonical(report))
        self.assertEqual(extra, {helper.ROOTED_PATH: ns["profile"].canonical(rooted)})
        adapter.verify_original.reset_mock()
        adapter.verify_original.side_effect = ValueError("invented original rejection")
        with self.assertRaisesRegex(ValueError, "invented original rejection"):
            ns["certify_once"]("NO_PROMOTION", lock, proof, Path("/unused"), bound, {}, {})
        adapter.verify_original.assert_called_once()
        direct.assert_not_called(); binder.assert_not_called()

    def test_wrapper_and_original_report_gates_cannot_be_relabelled(self):
        ns, adapter, lock, bound, report, rooted = self.fixture()
        for key, value in (("import_scope_adapter_sha256", "0" * 64), ("original_reader_sha256", "0" * 64),
                ("original_failed_continuation_run_id", 1), ("original_standalone_continuation_succeeded", True),
                ("panel_gate", "PASS"), ("promotion_allowed", True), ("new_model_calls", False),
                ("original_report_sha256", "0" * 64)):
            adapter.verify_original.return_value = dict(rooted, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                ns["certify_once"]("NO_PROMOTION", lock, Path("/invented"), Path("/unused"), bound, {}, {})
        for key, value in (("panel_gate", "PASS"), ("all_original_arrays_verified", False),
                           ("original_training_report_sha256", "0" * 64), ("replication_open_allowed", True)):
            bad_report = dict(report, **{key: value})
            adapter.verify_original.return_value = dict(rooted, original_report=bad_report,
                original_report_sha256=ns["profile"].sha(ns["profile"].canonical(bad_report)))
            with self.subTest(inner=key), self.assertRaises(ValueError):
                ns["certify_once"]("NO_PROMOTION", lock, Path("/invented"), Path("/unused"), bound, {}, {})

    def test_invented_complete_prepare_calls_adapter_once_and_failure_emits_no_complete(self):
        # Only authority/transport boundaries are fictional here. The pinned
        # prepare, archive hashing, 18-file layout, old report check and six-file
        # export run unchanged; no real artifact or scientific reader is used.
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                ns, adapter, _lock, bound, report, rooted = self.fixture()
                p = self.original.profile
                ns["ROOT"] = Path(temporary)
                cfg = config()
                members = {key: ("invented " + key + "\n").encode() for key in ns["HASH_FILES"]}
                cfg.update({key: p.sha(raw) for key, raw in members.items()})
                spec = dict(schema=self.original.evaluator.SCHEMA, panel="primary", scoring_allowed=True,
                    producer_manifest_sha256=cfg["producer_manifest_sha256"],
                    external_proof_binding_sha256=cfg["external_proof_binding_sha256"],
                    model_pt_sha256=bound["training_lock"]["training"]["model_pt_sha256"],
                    assessment_sha256=bound["assessment_sha256"])
                members["inputs_sha256"] = p.canonical(spec)
                cfg["inputs_sha256"] = p.sha(members["inputs_sha256"])
                for key, path in ns["HASH_FILES"].items():
                    target = ns["ROOT"] / path; target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(members[key])
                raw_result = p.canonical(dict(schema=spec["schema"], run_id=str(cfg["primary_run_id"]),
                    run_head=cfg["primary_run_head"], run_attempt=1, prepared_parent=cfg["primary_prepared_parent"],
                    registered_inputs_sha256=cfg["inputs_sha256"], producer_manifest_sha256=cfg["producer_manifest_sha256"],
                    implementation_sha256=cfg["evaluator_sha256"], model_pt_sha256=spec["model_pt_sha256"],
                    assessment_sha256=spec["assessment_sha256"], source_gate="SOURCE_GATE_PASS", source_shortfall={},
                    panel_gate="NO_PROMOTION"))

                def zip_bytes(files):
                    out = io.BytesIO()
                    with zipfile.ZipFile(out, "w") as archive:
                        for path, raw in files.items():
                            archive.writestr(path, raw)
                    return out.getvalue()

                payloads = {100: zip_bytes({"RESULT.json": raw_result}),
                    101: zip_bytes({"primary_producer/" + ns["HASH_FILES"][key]: raw for key, raw in members.items()}),
                    102: b"invented consumed"}
                inputs = {}
                for i, role in enumerate(sorted(ns["INPUT_ROLES"]), 200):
                    payloads[i] = ("invented input " + role).encode()
                    inputs[role] = dict(id=i, bytes=len(payloads[i]), sha256=p.sha(payloads[i]),
                                       run_id=123, run_head="1" * 40)
                run = dict(id=cfg["primary_run_id"], head_sha=cfg["primary_run_head"], run_attempt=1, event="push",
                    path=self.original.evaluator.WORKFLOW, head_branch=self.original.evaluator.BRANCH.removeprefix("refs/heads/"),
                    status="completed", conclusion="success", repository=dict(full_name=self.original.REPO, private=False))
                commit = dict(sha=cfg["primary_run_head"], parents=[dict(sha=cfg["primary_prepared_parent"])],
                              files=[dict(filename=self.original.evaluator.WORKFLOW)])
                items = [dict(id=i, name=name, size_in_bytes=len(payloads[i]), expired=False,
                    digest="sha256:" + p.sha(payloads[i]), workflow_run=dict(id=run["id"], head_sha=run["head_sha"]))
                    for i, name in enumerate(self.original.quality.artifact_names("primary", run["id"]).values(), 100)]
                endpoints = {"actions/runs/" + str(run["id"]): p.canonical(run),
                    "commits/" + run["head_sha"]: p.canonical(commit),
                    "actions/runs/" + str(run["id"]) + "/artifacts?per_page=100": p.canonical(dict(total_count=3, artifacts=items)),
                    **{"actions/artifacts/" + str(i) + "/zip": raw for i, raw in payloads.items()}}
                network = mock.Mock(side_effect=lambda endpoint: endpoints[endpoint])
                ns["evaluator"] = types.SimpleNamespace(**{**vars(self.original.evaluator), "gh": network})
                ns["validate_config"] = mock.Mock(return_value=cfg)
                ns["input_pins"] = mock.Mock(return_value=inputs)
                direct = mock.Mock(side_effect=AssertionError("no actual standalone reader"))
                ns["quality"] = types.SimpleNamespace(**{**vars(self.original.quality),
                    "validate_lock": mock.Mock(), "verify_quality_original": direct})
                original_bytes = {"invented/" + str(i) + ".json": p.canonical(dict(invented=i)) for i in range(41)}
                bound.update(files={path: dict(bytes=len(raw), sha256=p.sha(raw)) for path, raw in original_bytes.items()},
                    transport={path: dict(original_api_base64=base64.b64encode(raw).decode())
                               for path, raw in original_bytes.items()})
                ns["profile"] = types.SimpleNamespace(**{**vars(p), "binding": lambda panel, *_:
                    bound if panel == "primary" else dict(status="INERT_UNTIL_COMPLETE_ORIGINAL_B27_TRAIN")})

                def fetch(directory, files, transport):
                    self.assertEqual((files, transport), (bound["files"], bound["transport"]))
                    for name, raw in original_bytes.items():
                        target = directory / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)

                ns["original_proof"] = types.SimpleNamespace(fetch_files=mock.Mock(side_effect=fetch))
                ns["source_binder"] = types.SimpleNamespace(bind=mock.Mock(side_effect=AssertionError("no binder")))

                def verify(directory, lock, proof):
                    self.assertEqual(directory, proof / "primary-quality")
                    self.assertEqual(len(p.inventory(directory)), 18)
                    self.assertEqual(p.inventory(proof / "training-origin"), bound["files"])
                    if failure:
                        raise ValueError("invented full original rejection")
                    actual_report = dict(report, original_lock_sha256=p.sha(p.canonical(lock)),
                        original_result_sha256=lock["original_result_sha256"],
                        external_proof_binding_sha256=lock["external_proof_binding_sha256"])
                    return dict(rooted, original_report=actual_report, original_report_sha256=p.sha(p.canonical(actual_report)))

                adapter.verify_original.side_effect = verify
                work = Path(temporary) / "work"
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    if failure:
                        with self.assertRaisesRegex(ValueError, "invented full original rejection"):
                            ns["prepare"](cfg, work)
                    else:
                        ns["prepare"](cfg, work)
                adapter.verify_original.assert_called_once()
                ns["original_proof"].fetch_files.assert_called_once()
                direct.assert_not_called(); ns["source_binder"].bind.assert_not_called()
                self.assertEqual(len(network.call_args_list), len(endpoints))
                self.assertEqual(len({call.args[0] for call in network.call_args_list}), len(endpoints))
                self.assertEqual(len(p.inventory(work / "proof/primary-quality")), 18)
                self.assertEqual(len(p.inventory(work / "proof/training-origin")), 41)
                if failure:
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertFalse((work / "review/export").exists())
                else:
                    self.assertEqual(len(p.inventory(work / "review/export")), 6)
                    self.assertEqual(sum(line.startswith(ns["COMPLETE_PREFIX"]) for line in stdout.getvalue().splitlines()), 1)
    def test_exact_six_outputs_roundtrip_and_truthful_failed_continuation_identity(self):
        ns, _adapter, files = self.exports()
        preserved = {path: files[path] for path in (ns["LOCK_PATH"], ns["REPORT_PATH"], ns["RESULT_PATH"], helper.ROOTED_PATH)}
        records, manifest = ns["make_export"](files, "NO_PROMOTION")
        self.assertEqual(manifest["file_count"], 6)
        self.assertEqual(manifest["outcome"], "NO_PROMOTION")
        self.assertEqual(set(files), set(self.original.COMMON_PATHS) | {helper.ROOTED_PATH})
        self.assertTrue(all(files[path] == raw for path, raw in preserved.items()))
        for item in manifest["files"]:
            chunks = [r for r in records if r["path"] == item["path"]]
            self.assertEqual([r["index"] for r in chunks], list(range(item["chunks"])))
            self.assertTrue(all(len(r["base64"]) <= ns["CHUNK_CHARS"] for r in chunks))
            raw = base64.b64decode("".join(r["base64"] for r in chunks), validate=True)
            self.assertEqual(raw, files[item["path"]])
            self.assertEqual((len(raw), helper.sha(raw)), (item["bytes"], item["sha256"]))
        receipt = helper.document(files[ns["RECEIPT_PATH"]])
        self.assertEqual(receipt["certification_route"], "rooted_original_no_promotion_readback")
        self.assertEqual(receipt["original_failed_continuation"], helper.FAILED_CONTINUATION)
        self.assertIs(receipt["original_standalone_continuation_succeeded"], False)
        self.assertIs(receipt["scientific_execution_authorized"], False)
        inventory = helper.document(files[ns["INVENTORY_PATH"]])
        self.assertEqual(inventory["training_origin_files"], {"invented": "unchanged"})
        self.assertEqual(inventory["primary_quality_files"], {"invented": "unchanged"})
        self.assertEqual(inventory["review_files"], {path: dict(bytes=len(raw), sha256=helper.sha(raw))
            for path, raw in files.items() if path != ns["INVENTORY_PATH"]})

    def test_no_pass_replica_missing_or_oversized_export(self):
        ns, _adapter, original = self.exports()
        cases = [("PASS", original), ("STOP", original),
            ("NO_PROMOTION", {**original, self.original.REPLICA_PATH: b"x"}),
            ("NO_PROMOTION", {k: v for k, v in original.items() if k != helper.ROOTED_PATH}),
            ("NO_PROMOTION", {**original, ns["RESULT_PATH"]: b"x" * ns["MAX_EXPORT_BYTES"]})]
        for outcome, files in cases:
            with self.assertRaises(ValueError):
                ns["make_export"](copy.deepcopy(files), outcome)
        self.assertEqual(ns["MAX_EXPORT_BYTES"], 8_000_000)
        self.assertIs(ns["quality"], self.original.quality)
        self.assertIs(ns["original_proof"], self.original.original_proof)

    def test_workflow_has_read_only_permissions_and_two_independent_parent_slots(self):
        path = Path(__file__).resolve().parents[1] / helper.WORKFLOW
        raw = path.read_bytes()
        text = raw.decode()
        self.assertIn("permissions:\n  contents: read\n  actions: read\n", text)
        self.assertIn("persist-credentials: false", text)
        self.assertIn(helper.BRANCH, text)
        self.assertNotIn("workflow_dispatch:", text)
        self.assertNotIn("actions: write", text)
        self.assertNotIn("contents: write", text)
        self.assertIn("github.run_attempt == 1", text)
        self.assertIn("python -I -S -B tools/b27_primary_rooted_no_promotion_continuation.py", text)
        ns, _adapter = self.runtime()
        if helper.PARENT_TOKEN in text:
            self.assertEqual(text.count(helper.PARENT_TOKEN), 2)
            parent = "4" * 40
            ns["validate_workflow_activation"](raw, raw.replace(helper.PARENT_TOKEN.encode(), parent.encode()), parent)
        else:
            parent = re.search(r"PREPARED_PARENT: '([0-9a-f]{40})'", text).group(1)
            self.assertEqual(text.count(parent), 2)
            ns["validate_workflow_activation"](raw.replace(parent.encode(), helper.PARENT_TOKEN.encode()), raw, parent)


if __name__ == "__main__":
    unittest.main()
