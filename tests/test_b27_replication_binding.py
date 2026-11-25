"""Invented originals and mocked IO only; no real proof, model or network call."""
from __future__ import annotations

import base64
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import b27_replication_binding as helper

p = helper.profile


def config():
    return dict(schema=helper.CONFIG_SCHEMA, enabled=True, source_run_id=123,
        source_run_head="1" * 40, source_prepared_parent="2" * 40,
        **dict.fromkeys(helper.HASH_FILES, "a" * 64))


def artifact(ident, run=123, head="1" * 40, raw=b"z"):
    return dict(id=ident, run_id=run, run_head=head, bytes=len(raw), sha256=p.sha(raw))


def source_run():
    return dict(id=123, head_sha="1" * 40, run_attempt=1,
        head_branch=helper.evaluator.SOURCE_BRANCH, path=helper.evaluator.SOURCE_WORKFLOW,
        event="push", status="completed", conclusion="success",
        repository=dict(full_name=helper.REPO, private=False))


def source_commit():
    return dict(sha="1" * 40, parents=[dict(sha="2" * 40)],
        files=[dict(filename=helper.evaluator.SOURCE_WORKFLOW)],
        commit=dict(tree=dict(sha="b" * 40)))


def original_files(names, first_id):
    bodies, files, transport = {}, {}, {}
    for index, name in enumerate(sorted(names)):
        body = b"{}" if name.endswith(".json") else b"z"
        bodies[name] = body
        files[name] = dict(bytes=len(body), sha256=p.sha(body))
        transport[name] = (dict(original_api_base64=base64.b64encode(body).decode())
                           if name.endswith(".json") else dict(artifact_id=first_id + index))
    return bodies, files, transport


def binding_fixture():
    cfg = config()
    train_names = (
        ["trained/TRAIN_LIVE.json"]
        + ["trained/TRAIN_" + name + ".zip" for name in ("models", "producer", "receipts", "consumed")]
        + ["qualification/no-promotion/" + name for name in helper.PRIMARY_FILES]
        + ["qualification/training/fixture_" + str(index) + ".zip" for index in range(18)])
    train_bodies, train_files, train_transport = original_files(train_names, 10000)
    primary_bodies, primary_files, primary_transport = original_files(helper.PRIMARY_FILES, 20000)
    training = dict(run_id=32, run_head="3" * 40, artifacts={},
        model_pt_sha256={name: char * 64 for name, char in
                        (("B0", "a"), ("B9", "b"), ("control", "c"), ("challenger", "d"))})
    primary = dict(schema="english-v6-b27-primary-external-proof-binding-v1",
        panel="primary", status="BOUND_COMPLETE_ORIGINAL_TRAIN",
        scientific_execution_authorized=False, assessment_sha256="e" * 64,
        training_lock=dict(schema="english-v6-b27-original-training-lock-v1", training=training),
        training_report_sha256=helper.TRAIN_REPORT_SHA,
        files=train_files, transport=train_transport)
    primary_binding_raw = p.canonical(primary)
    primary_eval_raw = b"invented primary evaluator\n"
    producer_raw = p.canonical(dict(schema="english-v6-b27-primary-producer-v1", frozen=True))
    primary_spec = dict(schema="english-v6-b27-fresh-primary-four-frozen-models-v1",
        panel="primary", scoring_allowed=True, producer_manifest_sha256=p.sha(producer_raw),
        external_proof_binding_sha256=p.sha(primary_binding_raw),
        model_pt_sha256=training["model_pt_sha256"], assessment_sha256=primary["assessment_sha256"])
    spec_raw = p.canonical(primary_spec)
    lock = dict(schema=helper.quality.LOCK_SCHEMA, panel="primary",
        **{role: artifact(9000 + i, run=90, head="9" * 40)
           for i, role in enumerate(("result", "producer", "consumed"))},
        evaluator_sha256=p.sha(primary_eval_raw), inputs_sha256=p.sha(spec_raw),
        producer_manifest_sha256=p.sha(producer_raw), original_result_sha256="f" * 64,
        external_proof_binding_sha256=p.sha(primary_binding_raw))
    bound = dict(primary, schema="english-v6-b27-replication-external-proof-binding-v1",
        panel="replication", primary_quality_lock=lock, primary_quality_files=primary_files,
        primary_quality_transport=primary_transport, primary_quality_report_sha256="f" * 64)
    raw = dict(replication_evaluator_sha256=b"invented replica evaluator\n",
        replication_proof_binding_sha256=p.canonical(bound),
        primary_inputs_sha256=spec_raw, primary_producer_manifest_sha256=producer_raw)
    cfg.update({key: p.sha(value) for key, value in raw.items()})
    return dict(config=cfg, raw=raw, bound=bound, primary=primary,
        primary_eval_raw=primary_eval_raw, primary_binding_raw=primary_binding_raw,
        train_bodies=train_bodies, primary_bodies=primary_bodies)


def screen_document(cfg, bound, gate="SOURCE_GATE_PASS"):
    return dict(schema=helper.evaluator.SCREEN_SCHEMA, panel="replication",
        source_gate=gate, source_shortfall={}, assessment_sha256=bound["assessment_sha256"],
        source_execution=dict(run_id=str(cfg["source_run_id"]), run_head=cfg["source_run_head"],
            prepared_parent=cfg["source_prepared_parent"], run_attempt=1,
            run_branch="refs/heads/" + helper.evaluator.SOURCE_BRANCH),
        fixture="Unicode \u6a21\u578b \u2705")


def zip_bytes(members):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0)), raw)
    return output.getvalue()


def write_generated(output, cfg, bound, pins, screen_raw):
    producer_raw = p.canonical(dict(schema="english-v6-b27-replication-producer-v1", frozen=True,
                                   fixture="invented complete producer"))
    spec = dict(schema=helper.evaluator.SCHEMA, panel="replication", scoring_allowed=True,
        primary_quality_lock=bound["primary_quality_lock"], original_primary_pass_required=True,
        replication_opened=True, external_proof_binding_sha256=cfg["replication_proof_binding_sha256"],
        producer_manifest_sha256=p.sha(producer_raw),
        model_pt_sha256=bound["training_lock"]["training"]["model_pt_sha256"],
        inputs={role: pins[role] for role in ("B0", "B9", "pair", "source", "screen")},
        native_inputs={role: pins["native_" + role] for role in helper.binder.PROVIDERS},
        upstream_producers={role: pins[role + "_producer"] for role in ("train", "source")})
    spec_raw = p.canonical(spec)
    audit = dict(schema="english-v6-b27-original-quality-binding-audit-v1", panel="replication",
        status="BOUND_ORIGINAL_INPUTS_FOR_REVIEWED_ONCE_QUALITY",
        evaluator_sha256=cfg["replication_evaluator_sha256"],
        registration_sha256=p.sha(spec_raw), producer_manifest_sha256=p.sha(producer_raw),
        source_audit=dict(screen_sha256=p.sha(screen_raw)),
        full_original_training_report_sha256=helper.TRAIN_REPORT_SHA,
        input_artifacts=pins, model_pt_sha256=spec["model_pt_sha256"],
        all_original_screens_recomputed=True, every_actual_upstream_producer_member_preserved=True,
        execution_started=False, promotion_allowed=False, product_released=False,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0, remote_mutations=0)
    output.mkdir()
    (output / "automation").mkdir()
    (output / helper.EXPORT_PATHS[0]).write_bytes(spec_raw)
    (output / helper.EXPORT_PATHS[1]).write_bytes(producer_raw)
    (output / "BINDING_AUDIT.json").write_bytes(p.canonical(audit))
    return audit


class EntryContracts(unittest.TestCase):
    def test_inert_configuration_precedes_env_git_network_and_mkdir(self):
        inert = dict(schema=helper.CONFIG_SCHEMA, enabled=False,
                     **dict.fromkeys(helper.CONFIG_PINS))
        class UnreadableEnvironment(dict):
            def __getitem__(self, key):
                raise AssertionError("environment accessed before disabled rejection")
            def get(self, *args):
                raise AssertionError("environment accessed before disabled rejection")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = root / "inert.json"
            settings.write_bytes(p.canonical(inert))
            work = root / "never-created"
            with mock.patch.object(helper, "CONFIG", settings), \
                 mock.patch.object(helper.os, "environ", UnreadableEnvironment()), \
                 mock.patch.object(helper.subprocess, "run") as child, \
                 mock.patch.object(helper.evaluator, "gh") as network, \
                 mock.patch.object(helper, "check_checkout") as checkout, \
                 mock.patch.object(helper, "fetch_proofs") as fetch, \
                 mock.patch.object(helper.binder, "bind") as bind, \
                 mock.patch.object(Path, "mkdir") as mkdir:
                with self.assertRaisesRegex(ValueError, "INERT"):
                    helper.main()
                with self.assertRaisesRegex(ValueError, "INERT"):
                    helper.prepare(inert, work)
            for call in (child, network, checkout, fetch, bind, mkdir):
                call.assert_not_called()
            self.assertFalse(work.exists())

    def test_exact_config_has_no_runtime_override(self):
        self.assertEqual(helper.validate_config(config()), config())
        current = p.document(helper.CONFIG.read_bytes())
        if current["enabled"] is False:
            self.assertTrue(all(current[key] is None for key in helper.CONFIG_PINS))
            with self.assertRaisesRegex(ValueError, "INERT"):
                helper.validate_config(current)
        else:
            helper.validate_config(current)
        for key, value in (
            ("enabled", 1), ("source_run_id", True), ("source_run_id", 0),
            ("source_run_head", "x" * 40), ("source_prepared_parent", "1" * 40),
            ("primary_inputs_sha256", None), ("replication_proof_binding_sha256", "A" * 64)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                helper.validate_config(dict(config(), **{key: value}))
        with self.assertRaises(ValueError):
            helper.validate_config(dict(config(), threshold=0.0))
        with self.assertRaises(ValueError):
            helper.validate_config(dict(config(), enabled=False))

    def test_exact_workflow_only_entry_and_two_substitutions(self):
        parent, head = "4" * 40, "5" * 40
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=helper.REPO,
            GITHUB_EVENT_NAME="push", GITHUB_REF=helper.BRANCH, GITHUB_RUN_ATTEMPT="1",
            GITHUB_SHA=head, PREPARED_PARENT=parent)
        event = dict(before=parent, after=head, ref=helper.BRANCH, forced=False, deleted=False,
            repository=dict(full_name=helper.REPO, private=False))
        helper.validate_entry(env, event, head, parent, [helper.WORKFLOW])
        for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_REF", helper.evaluator.BRANCH),
                           ("PREPARED_PARENT", "6" * 40), ("GITHUB_EVENT_NAME", "pull_request")):
            with self.assertRaises(ValueError):
                helper.validate_entry(dict(env, **{key: value}), event, head, parent, [helper.WORKFLOW])
        with self.assertRaises(ValueError):
            helper.validate_entry(env, event, head, parent, [helper.WORKFLOW, "automation/changed.py"])
        for key in ("forced", "deleted"):
            with self.assertRaises(ValueError):
                helper.validate_entry(env, dict(event, **{key: True}), head, parent, [helper.WORKFLOW])
        before = (helper.PARENT_TOKEN + "\n" + helper.PARENT_TOKEN).encode()
        active = before.replace(helper.PARENT_TOKEN.encode(), parent.encode())
        helper.validate_workflow_activation(before, active, parent)
        for old, new in ((before, active + b"x"), (before, before),
                         (before + helper.PARENT_TOKEN.encode(), active)):
            with self.assertRaises(ValueError):
                helper.validate_workflow_activation(old, new, parent)

    def test_source_identity_terminal_status_and_actual_parent(self):
        cfg, run, commit = config(), source_run(), source_commit()
        helper.validate_source_run(cfg, run)
        helper.validate_source_commit(cfg, commit)
        for key, value in (
            ("status", "in_progress"), ("conclusion", None), ("conclusion", "failure"),
            ("run_attempt", 2), ("run_attempt", True), ("id", 124), ("head_sha", "0" * 40),
            ("head_branch", "main"), ("path", helper.WORKFLOW), ("event", "workflow_dispatch")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                helper.validate_source_run(cfg, dict(run, **{key: value}))
        private = copy.deepcopy(run); private["repository"]["private"] = True
        with self.assertRaises(ValueError):
            helper.validate_source_run(cfg, private)
        for changed in (
            dict(commit, parents=[dict(sha="0" * 40)]),
            dict(commit, parents=[dict(sha="2" * 40), dict(sha="3" * 40)]),
            dict(commit, files=[dict(filename="automation/changed.py")])):
            with self.assertRaises(ValueError):
                helper.validate_source_commit(cfg, changed)

    def test_nonfinal_source_stops_before_directory_proof_and_binder(self):
        cfg = config()
        run = dict(source_run(), status="in_progress", conclusion=None)
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary).resolve() / "fresh"
            with mock.patch.object(helper, "check_checkout", return_value={}), \
                 mock.patch.object(helper.evaluator, "gh", return_value=p.canonical(run)) as network, \
                 mock.patch.object(helper, "load_bound") as bound, \
                 mock.patch.object(helper, "fetch_proofs") as fetch, \
                 mock.patch.object(helper.binder, "bind") as bind:
                with self.assertRaises(ValueError):
                    helper.prepare(cfg, work)
            network.assert_called_once_with("actions/runs/123")
            bound.assert_not_called(); fetch.assert_not_called(); bind.assert_not_called()
            self.assertFalse(work.exists())

    def test_complete_original_api_inventory_tree_and_source_history(self):
        cfg, run, commit = config(), source_run(), source_commit()
        inventory = dict(total_count=1, artifacts=[dict(id=1)])
        tree = dict(sha="b" * 40, truncated=False, tree=[])
        helper.validate_api_group("source", run, commit, inventory, 123, "1" * 40, tree)
        helper.validate_history(cfg, dict(total_count=1, workflow_runs=[run]))
        cases = [
            (dict(inventory, total_count=2), commit, tree),
            (inventory, dict(commit, files=[dict(filename=str(i)) for i in range(100)]), tree),
            (inventory, commit, dict(tree, truncated=True)),
            (inventory, commit, dict(tree, sha="1" * 40)),
        ]
        for inv, com, tr in cases:
            with self.assertRaises(ValueError):
                helper.validate_api_group("source", run, com, inv, 123, "1" * 40, tr)
        for history in (dict(total_count=2, workflow_runs=[run]),
                        dict(total_count=2, workflow_runs=[run, run]),
                        dict(total_count=1, workflow_runs=[dict(run, id=99)])):
            with self.assertRaises(ValueError):
                helper.validate_history(cfg, history)


class OriginalProofContracts(unittest.TestCase):
    def validate(self, fixture):
        helper.validate_binding_links(fixture["config"], fixture["raw"],
            fixture["bound"], fixture["primary"], fixture["primary_eval_raw"],
            fixture["primary_binding_raw"])

    def test_exact_train41_primary18_and_lock_file_linkage(self):
        fixture = binding_fixture()
        self.validate(fixture)
        self.assertEqual(len(fixture["bound"]["files"]), 41)
        self.assertEqual(len(fixture["bound"]["primary_quality_files"]), 18)
        cases = []
        bad = copy.deepcopy(fixture); bad["raw"]["primary_inputs_sha256"] += b"\n"; cases.append(bad)
        bad = copy.deepcopy(fixture); bad["primary_eval_raw"] += b"x"; cases.append(bad)
        bad = copy.deepcopy(fixture); bad["primary_binding_raw"] += b" "; cases.append(bad)
        for key, value in (("inputs_sha256", "0" * 64), ("producer_manifest_sha256", "0" * 64)):
            bad = copy.deepcopy(fixture)
            bad["bound"]["primary_quality_lock"][key] = value
            bad["raw"]["replication_proof_binding_sha256"] = p.canonical(bad["bound"])
            bad["config"]["replication_proof_binding_sha256"] = p.sha(bad["raw"]["replication_proof_binding_sha256"])
            cases.append(bad)
        bad = copy.deepcopy(fixture)
        bad["bound"]["primary_quality_files"].pop("inputs/native_qwen.zip")
        bad["raw"]["replication_proof_binding_sha256"] = p.canonical(bad["bound"])
        bad["config"]["replication_proof_binding_sha256"] = p.sha(bad["raw"]["replication_proof_binding_sha256"])
        cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):
                self.validate(bad)

    def test_screen_is_exact_raw_source_gate_not_model_quality(self):
        fixture = binding_fixture()
        cfg, bound = fixture["config"], fixture["bound"]
        screen = screen_document(cfg, bound)
        raw = json.dumps(screen, indent=2, ensure_ascii=False).encode() + b"\n"
        self.assertEqual(helper.validate_screen(cfg, raw, bound), screen)
        self.assertNotEqual(p.sha(raw), p.sha(p.canonical(screen)))
        for key, value in (("source_gate", "INSUFFICIENT_SOURCE"), ("panel", "primary"),
                           ("schema", "wrong"), ("source_shortfall", {"missing": 1}),
                           ("assessment_sha256", "0" * 64)):
            with self.assertRaises(ValueError):
                helper.validate_screen(cfg, p.canonical(dict(screen, **{key: value})), bound)
        for key, value in (("run_id", "124"), ("run_head", "0" * 40),
                           ("prepared_parent", "0" * 40), ("run_attempt", 2), ("run_branch", helper.BRANCH)):
            bad = copy.deepcopy(screen); bad["source_execution"][key] = value
            with self.assertRaises(ValueError):
                helper.validate_screen(cfg, p.canonical(bad), bound)

    def test_only_two_fetch_files_calls_and_no_complete_reader(self):
        fixture = binding_fixture()
        bound = fixture["bound"]
        with tempfile.TemporaryDirectory() as temporary:
            proof = Path(temporary).resolve() / "proof"
            with mock.patch.object(helper.original_proof, "fetch_files") as fetch, \
                 mock.patch.object(helper.original_proof, "fetch", side_effect=AssertionError("no fetch CLI")) as full, \
                 mock.patch.object(helper.profile, "require_proof", side_effect=AssertionError("no pre-reader")) as reader:
                helper.fetch_proofs(proof, bound)
            self.assertEqual(fetch.call_args_list, [
                mock.call(proof / "training-origin", bound["files"], bound["transport"]),
                mock.call(proof / "primary-quality", bound["primary_quality_files"], bound["primary_quality_transport"])])
            full.assert_not_called(); reader.assert_not_called()

    def test_output_binding_requires_same_model_primary_lock_and_raw_hashes(self):
        fixture = binding_fixture()
        cfg, bound = fixture["config"], fixture["bound"]
        screen_raw = json.dumps(screen_document(cfg, bound), indent=2).encode() + b"\n"
        pins = {role: artifact(100 + i) for i, role in enumerate(helper.binder.ROLES)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "valid"
            audit = write_generated(output, cfg, bound, pins, screen_raw)
            exported = helper.verify_outputs(cfg, output, audit, screen_raw, bound, pins)
            self.assertEqual(set(exported), set(helper.EXPORT_PATHS[:4]))
            self.assertEqual(exported[helper.EXPORT_PATHS[3]], screen_raw)
            cases = [
                ("execution_started", True), ("new_model_calls", 1), ("new_source_calls", False),
                ("all_original_screens_recomputed", False), ("promotion_allowed", True),
                ("evaluator_sha256", "0" * 64), ("full_original_training_report_sha256", "0" * 64),
            ]
            for index, (key, value) in enumerate(cases):
                out = root / str(index)
                bad = write_generated(out, cfg, bound, pins, screen_raw)
                bad[key] = value
                (out / "BINDING_AUDIT.json").write_bytes(p.canonical(bad))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    helper.verify_outputs(cfg, out, bad, screen_raw, bound, pins)
            with self.assertRaises(ValueError):
                helper.verify_outputs(cfg, output, audit, p.canonical(p.document(screen_raw)), bound, pins)
            path = output / helper.EXPORT_PATHS[0]
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                helper.verify_outputs(cfg, output, audit, screen_raw, bound, pins)


class TransportFixture:
    def __init__(self, gate="SOURCE_GATE_PASS"):
        fixture = binding_fixture()
        self.config, self.bound = fixture["config"], fixture["bound"]
        self.raw = fixture["raw"]
        self.bodies = {"training-origin": fixture["train_bodies"],
                       "primary-quality": fixture["primary_bodies"]}
        self.screen_raw = json.dumps(screen_document(self.config, self.bound, gate),
                                     ensure_ascii=False, indent=2).encode() + b"\n"
        self.archives = {role: zip_bytes({"SCREEN.json": self.screen_raw} if role == "screen"
                                        else {"INVENTED.txt": role.encode()})
                         for role in helper.binder.ROLES}
        origins = {"B0": (10, "4" * 40), "B9": (20, "5" * 40),
                   "train": (32, "3" * 40), "source": (123, "1" * 40)}
        self.pins = {}
        for index, role in enumerate(helper.binder.ROLES):
            group = role if role in ("B0", "B9") else "train" if role in ("pair", "train_producer") else "source"
            run, head = origins[group]
            self.pins[role] = artifact(100 + index, run, head, self.archives[role])
        training = self.bound["training_lock"]["training"]
        training["artifacts"] = {
            name: {key: self.pins[role][key] for key in ("id", "bytes", "sha256")}
            for role, name in (("pair", "models"), ("train_producer", "producer"))}
        self.baselines = {role: self.pins[role] for role in ("B0", "B9")}
        for directory, file_key, transport_key in (
            ("training-origin", "files", "transport"),
            ("primary-quality", "primary_quality_files", "primary_quality_transport")):
            for role in ("B0", "B9", "pair", "train_producer"):
                if directory == "training-origin":
                    name = ({"pair": "trained/TRAIN_models.zip", "train_producer": "trained/TRAIN_producer.zip"}
                            .get(role, "qualification/no-promotion/inputs/" + role + ".zip"))
                else:
                    name = "inputs/" + role + ".zip"
                raw = self.archives[role]
                self.bodies[directory][name] = raw
                self.bound[file_key][name] = dict(bytes=len(raw), sha256=p.sha(raw))
                self.bound[transport_key][name] = dict(artifact_id=self.pins[role]["id"])
        self.api = {}
        for name, (run_id, head) in origins.items():
            run = dict(source_run(), id=run_id, head_sha=head)
            if name == "train":
                run.update(head_branch=helper.evaluator.TRAIN_BRANCH, path=helper.evaluator.TRAIN_WORKFLOW)
            commit = dict(sha=head, parents=[dict(sha="2" * 40)],
                files=[dict(filename=helper.evaluator.SOURCE_WORKFLOW if name == "source" else "fixture.txt")],
                commit=dict(tree=dict(sha=("a" if name == "train" else "b") * 40)))
            items = []
            for role, pin in self.pins.items():
                if pin["run_id"] != run_id:
                    continue
                artifact_name = ("invented-" + role if role in ("B0", "B9")
                                 else helper.binder.artifact_name(role, run_id, "replication"))
                items.append(dict(id=pin["id"], name=artifact_name, size_in_bytes=pin["bytes"],
                    digest="sha256:" + pin["sha256"], expired=False,
                    workflow_run=dict(id=run_id, head_sha=head)))
            self.api["actions/runs/" + str(run_id)] = p.canonical(run)
            self.api["commits/" + head + "?per_page=100"] = p.canonical(commit)
            self.api["actions/runs/" + str(run_id) + "/artifacts?per_page=100"] = p.canonical(
                dict(total_count=len(items), artifacts=items))
            if name in ("train", "source"):
                tree = commit["commit"]["tree"]["sha"]
                self.api["git/trees/" + tree + "?recursive=1"] = p.canonical(
                    dict(sha=tree, truncated=False, tree=[]))
        query = helper.urlencode(dict(branch=helper.evaluator.SOURCE_BRANCH, event="push", per_page=100))
        self.api["actions/workflows/" + Path(helper.evaluator.SOURCE_WORKFLOW).name + "/runs?" + query] = \
            p.canonical(dict(total_count=1, workflow_runs=[source_run()]))
        for role, pin in self.pins.items():
            self.api["actions/artifacts/" + str(pin["id"]) + "/zip"] = self.archives[role]
        self.events = []

    def gh(self, endpoint):
        self.events.append(("gh", endpoint))
        if endpoint not in self.api:
            raise AssertionError("unexpected endpoint " + endpoint)
        return self.api[endpoint]

    def fetch(self, directory, files, transport):
        self.events.append(("fetch_files", directory.name))
        directory.mkdir()
        self.assert_equal(set(files), set(self.bodies[directory.name]))
        for name, raw in self.bodies[directory.name].items():
            self.assert_equal(files[name], dict(bytes=len(raw), sha256=p.sha(raw)))
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)

    @staticmethod
    def assert_equal(left, right):
        if left != right:
            raise AssertionError("invented fixture identity mismatch")

    def bind(self, panel, bundle, proof, output):
        self.events.append(("binder", panel))
        self.assert_equal(panel, "replication")
        self.assert_equal(len(p.inventory(proof / "training-origin")), 41)
        self.assert_equal(len(p.inventory(proof / "primary-quality")), 18)
        self.assert_equal(len(p.inventory(bundle.parent / "archives")), 12)
        return write_generated(output, self.config, self.bound, self.pins, self.screen_raw)

    def patches(self, stack, bind=None):
        stack.enter_context(mock.patch.object(helper, "check_checkout", return_value=self.raw))
        stack.enter_context(mock.patch.object(helper, "load_bound", return_value=self.bound))
        stack.enter_context(mock.patch.object(helper.evaluator, "BASELINES", self.baselines))
        network = stack.enter_context(mock.patch.object(helper.evaluator, "gh", side_effect=self.gh))
        fetch = stack.enter_context(mock.patch.object(helper.original_proof, "fetch_files", side_effect=self.fetch))
        binder = stack.enter_context(mock.patch.object(helper.binder, "bind", side_effect=bind or self.bind))
        for owner, name in ((helper.original_proof, "fetch"), (helper.profile, "require_proof"),
                            (helper.quality, "verify_quality_original"), (helper.subprocess, "run")):
            stack.enter_context(mock.patch.object(owner, name, side_effect=AssertionError("forbidden real reader/IO")))
        return network, fetch, binder


class OrchestrationContracts(unittest.TestCase):
    def test_small_screen_then_two_whole_proofs_then_one_original_binder(self):
        fixture = TransportFixture()
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            work = Path(temporary).resolve() / "work"
            network, fetch, binder = fixture.patches(stack)
            capture = io.StringIO()
            with redirect_stdout(capture):
                helper.prepare(fixture.config, work)
            self.assertEqual(fetch.call_count, 2)
            binder.assert_called_once_with("replication", work / "bundle/BUNDLE.json",
                                           work / "proof", work / "bound")
            screen_request = ("gh", "actions/artifacts/" + str(fixture.pins["screen"]["id"]) + "/zip")
            self.assertLess(fixture.events.index(screen_request), fixture.events.index(("fetch_files", "training-origin")))
            self.assertLess(fixture.events.index(("fetch_files", "primary-quality")), fixture.events.index(("binder", "replication")))
            for role in ("B0", "B9", "pair", "train_producer"):
                request = ("gh", "actions/artifacts/" + str(fixture.pins[role]["id"]) + "/zip")
                self.assertNotIn(request, fixture.events)
            self.assertEqual(fixture.events.count(screen_request), 1)
            exported = p.inventory(work / "review/export")
            self.assertEqual(set(exported), set(helper.EXPORT_PATHS))
            self.assertEqual((work / "review/export" / helper.EXPORT_PATHS[3]).read_bytes(), fixture.screen_raw)
            self.assertEqual(len(p.inventory(work / "review/raw/api")), 15)
            inventory = p.document((work / "review/export" / helper.EXPORT_PATHS[4]).read_bytes())
            self.assertEqual(inventory["primary_quality_files"], fixture.bound["primary_quality_files"])
            self.assertFalse(inventory["replication_quality_passed"])
            self.assertFalse(inventory["scientific_execution_authorized"])
            lines = capture.getvalue().splitlines()
            self.assertEqual(sum(line.startswith(helper.COMPLETE_PREFIX) for line in lines), 1)
            self.assertTrue(lines[-1].startswith(helper.COMPLETE_PREFIX))
            manifest = p.document(lines[-1][len(helper.COMPLETE_PREFIX):].encode())
            self.assertEqual(manifest["file_count"], 5)
            self.assertEqual(manifest["total_bytes"], sum(pin["bytes"] for pin in exported.values()))

    def test_failed_raw_source_gate_never_downloads_large_proof_or_calls_binder(self):
        fixture = TransportFixture(gate="INSUFFICIENT_SOURCE")
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            work = Path(temporary).resolve() / "work"
            _, fetch, binder = fixture.patches(stack)
            capture = io.StringIO()
            with redirect_stdout(capture), self.assertRaises(ValueError):
                helper.prepare(fixture.config, work)
            fetch.assert_not_called(); binder.assert_not_called()
            self.assertFalse((work / "proof").exists())
            self.assertFalse((work / "review/export").exists())
            self.assertEqual((work / "review/raw/SCREEN.json").read_bytes(), fixture.screen_raw)
            self.assertEqual(capture.getvalue(), "")

    def test_truncated_api_never_transports_any_zip(self):
        fixture = TransportFixture()
        key = "actions/runs/123/artifacts?per_page=100"
        inventory = p.document(fixture.api[key]); inventory["total_count"] += 1
        fixture.api[key] = p.canonical(inventory)
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            work = Path(temporary).resolve() / "work"
            _, fetch, binder = fixture.patches(stack)
            with self.assertRaises(ValueError):
                helper.prepare(fixture.config, work)
            fetch.assert_not_called(); binder.assert_not_called()
            self.assertFalse(any(kind == "gh" and endpoint.endswith("/zip") for kind, endpoint in fixture.events))

    def test_binder_failure_never_retries_or_exports_partial_registration(self):
        fixture = TransportFixture()
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            work = Path(temporary).resolve() / "work"
            _, fetch, binder = fixture.patches(stack, bind=mock.Mock(side_effect=ValueError("invented invalid proof")))
            capture = io.StringIO()
            with redirect_stdout(capture), self.assertRaisesRegex(ValueError, "invented invalid proof"):
                helper.prepare(fixture.config, work)
            binder.assert_called_once()
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual(capture.getvalue(), "")
            self.assertFalse((work / "review/export").exists())
            self.assertEqual(len(p.inventory(work / "review/raw/api")), 15)
            self.assertEqual(len(p.inventory(work / "proof/training-origin")), 41)
            self.assertEqual(len(p.inventory(work / "proof/primary-quality")), 18)


class ExportContracts(unittest.TestCase):
    def test_five_original_files_multichunk_unicode_roundtrip(self):
        files = {path: ("\u6a21\u578b \u2705 \U0001f9ea\n" * 4000 + str(i)).encode()
                 for i, path in enumerate(helper.EXPORT_PATHS)}
        records, manifest = helper.make_export(files)
        self.assertEqual(manifest["file_count"], 5)
        self.assertEqual(manifest["total_bytes"], sum(map(len, files.values())))
        for pin in manifest["files"]:
            chunks = [row for row in records if row["path"] == pin["path"]]
            self.assertGreater(pin["chunks"], 1)
            self.assertEqual([row["index"] for row in chunks], list(range(pin["chunks"])))
            self.assertTrue(all(len(row["base64"]) <= 32768 for row in chunks))
            self.assertTrue(all({key: row[key] for key in pin} == pin for row in chunks))
            raw = base64.b64decode("".join(row["base64"] for row in chunks), validate=True)
            self.assertEqual(raw, files[pin["path"]])
            self.assertEqual(len(raw), pin["bytes"]); self.assertEqual(p.sha(raw), pin["sha256"])
        self.assertEqual(helper.CHUNK_PREFIX, "B27_REPLICATION_BINDING_CHUNK ")
        self.assertEqual(helper.COMPLETE_PREFIX, "B27_REPLICATION_BINDING_COMPLETE ")

    def test_export_rejects_extra_missing_empty_and_oversize_metadata(self):
        files = dict.fromkeys(helper.EXPORT_PATHS, b"x")
        for bad in ({**files, "extra.json": b"x"},
                    {key: value for key, value in files.items() if key != helper.EXPORT_PATHS[0]},
                    {**files, helper.EXPORT_PATHS[0]: b""},
                    {**files, helper.EXPORT_PATHS[0]: b"x" * helper.MAX_EXPORT_BYTES}):
            with self.assertRaises(ValueError):
                helper.make_export(bad)


if __name__ == "__main__":
    unittest.main()
