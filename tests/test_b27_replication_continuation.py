"""Invented bytes and API metadata only; every full reader/network is mocked."""
from __future__ import annotations

import base64
import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import b27_replication_continuation as h

p = h.profile


def config():
    return dict(schema=h.CONFIG_SCHEMA, enabled=True, replication_run_id=123,
        replication_run_head="1" * 40, replication_prepared_parent="2" * 40,
        **{key: "a" * 64 for key in h.HASH_FILES})


def archive(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as value:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            value.writestr(info, raw)
    return buffer.getvalue()


def byte_pin(raw):
    return dict(bytes=len(raw), sha256=p.sha(raw))


def pin(ident, raw, run=123, head="1" * 40):
    return dict(id=ident, run_id=run, run_head=head, **byte_pin(raw))


def generic_lock_check(lock):
    p.need(set(lock) == h.quality.LOCK_FIELDS and lock["schema"] == h.quality.LOCK_SCHEMA,
           "fixture uses exact real lock fields")
    for role in h.OUTPUT_ROLES:
        h.quality.valid_artifact(lock[role], panel=lock["panel"], role=role)
    for key in (*h.HASH_FILES, "original_result_sha256"):
        p.need(p.valid_hash(lock[key]), "fixture digest")


def report(panel, gate, lock, bound, pair):
    result = dict(schema=h.quality.REPORT_SCHEMA, panel=panel, panel_gate=gate,
        original_lock_sha256=p.sha(p.canonical(lock)),
        original_result_sha256=lock["original_result_sha256"],
        external_proof_binding_sha256=lock["external_proof_binding_sha256"],
        original_training_report_sha256=bound["training_report_sha256"],
        model_pt_sha256=bound["training_lock"]["training"]["model_pt_sha256"],
        assessment_sha256=bound["assessment_sha256"], pair_pin=pair,
        replication_open_allowed=panel == "primary" and gate == "PASS",
        candidate_engineering_validation_allowed=panel == "replication" and gate == "PASS",
        automatic_next_assessment_allowed=False, promotion_allowed=False, product_released=False,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0)
    result.update(dict.fromkeys((
        "all_original_arrays_verified", "original_source_admission_recomputed",
        "original_producer_git_verified", "all_original_training_proof_verified",
        "complete_original_Q1_failure_verified", "old_stops_and_reserves_preserved"), True))
    return result


class Fixture:
    """Complete 41+18+18 shape, entirely fabricated content and artifact IDs."""
    def __init__(self, gate="PASS"):
        self.cfg = config()
        self.gate = gate
        self.events, self.requests = [], []
        self.blobs = {}
        self.models = dict(B0="a" * 64, B9="b" * 64, control="c" * 64, challenger="d" * 64)
        self.inert_doc = p.document((h.ROOT / h.ENGINEERING_SPEC).read_bytes())
        self.shared = {}
        for i, role in enumerate(("B0", "B9", "pair", "train_producer")):
            raw = archive({"INVENTED.txt": ("shared " + role).encode()})
            self.blobs[1000 + i] = raw
            self.shared[role] = pin(1000 + i, raw, run=10, head="a" * 40)

        self.train_bytes = {}
        for path in sorted(h.QUALITY_FILES):
            for prefix in ("qualification/no-promotion/", "qualification/training/"):
                body = b'{"invented":true}' if path.endswith(".json") else archive({"INVENTED.txt": path.encode()})
                self.train_bytes[prefix + path] = body
        self.train_bytes.update({
            "trained/TRAIN_LIVE.json": b'{"invented":"training"}',
            "trained/TRAIN_models.zip": self.blobs[self.shared["pair"]["id"]],
            "trained/TRAIN_producer.zip": self.blobs[self.shared["train_producer"]["id"]],
            "trained/TRAIN_receipts.zip": archive({"INVENTED.txt": b"receipts"}),
            "trained/TRAIN_consumed.zip": archive({"INVENTED.txt": b"consumed"}),
        })
        for role in ("B0", "B9"):
            self.train_bytes["qualification/training/inputs/" + role + ".zip"] = self.blobs[self.shared[role]["id"]]
        train_files = {path: byte_pin(raw) for path, raw in self.train_bytes.items()}
        train_transport = {}
        for i, (path, raw) in enumerate(sorted(self.train_bytes.items())):
            if path.endswith(".json"):
                train_transport[path] = dict(original_api_base64=base64.b64encode(raw).decode("ascii"))
            else:
                reused = next((ident for ident, body in self.blobs.items() if body == raw), None)
                ident = reused if reused is not None else 3000 + i
                self.blobs[ident] = raw
                train_transport[path] = dict(artifact_id=ident)
        self.primary_bound = dict(schema="english-v6-b27-primary-external-proof-binding-v1",
            panel="primary", status="BOUND_COMPLETE_ORIGINAL_TRAIN",
            scientific_execution_authorized=False, assessment_sha256="e" * 64,
            training_reader_sha256="f" * 64, training_report_sha256=h.TRAIN_REPORT_SHA,
            training_lock=dict(training=dict(run_id=10, run_head="a" * 40,
                model_pt_sha256=self.models,
                artifacts=dict(models={k: self.shared["pair"][k] for k in ("id", "bytes", "sha256")},
                    producer={k: self.shared["train_producer"][k] for k in ("id", "bytes", "sha256")}))),
            files=train_files, transport=train_transport)
        self.primary_inputs = self.make_inputs(4000, 122, "5" * 40)
        primary_producer = p.canonical(dict(schema="english-v6-b27-primary-producer-v1", frozen=True))
        self.primary_raw = dict(
            evaluator_sha256=b"INVENTED primary evaluator\n",
            producer_manifest_sha256=primary_producer,
            external_proof_binding_sha256=p.canonical(self.primary_bound))
        primary_spec = self.spec("primary", self.primary_inputs,
            p.sha(primary_producer), p.sha(self.primary_raw["external_proof_binding_sha256"]))
        self.primary_raw["inputs_sha256"] = p.canonical(primary_spec)
        primary_result = b'{ "invented": "PRIMARY", "panel_gate": "PASS" }\n'
        original_bytes = {
            "result": archive({"RESULT.json": primary_result}),
            "producer": archive({"primary_producer/" + path.replace("_replication_", "_primary_"):
                                  self.primary_raw[key] for key, path in h.HASH_FILES.items()}),
            "consumed": archive({"CONSUMPTION.json": b'{"invented":true}'}),
        }
        primary_outputs = {}
        for i, role in enumerate(h.OUTPUT_ROLES):
            ident = 5000 + i
            self.blobs[ident] = original_bytes[role]
            primary_outputs[role] = pin(ident, original_bytes[role], run=122, head="5" * 40)
        self.primary_lock = dict(schema=h.quality.LOCK_SCHEMA, panel="primary", **primary_outputs,
            **{key: p.sha(raw) for key, raw in self.primary_raw.items()},
            original_result_sha256=p.sha(primary_result))
        self.primary_bytes = {
            "api/run.json": b'{"invented":"saved primary run"}',
            "api/commit.json": b'{"invented":"saved primary commit"}',
            "api/artifacts.json": b'{"invented":"saved primary artifacts"}',
            **{"originals/" + role + ".zip": body for role, body in original_bytes.items()},
            **{"inputs/" + role + ".zip": self.blobs[value["id"]] for role, value in self.primary_inputs.items()},
        }
        primary_files, primary_transport = h.quality_layout(
            {key: self.primary_bytes["api/" + key + ".json"] for key in ("run", "commit", "artifacts")},
            primary_outputs, self.primary_inputs)
        self.bound = dict(copy.deepcopy(self.primary_bound),
            schema="english-v6-b27-replication-external-proof-binding-v1", panel="replication",
            primary_quality_lock=self.primary_lock, primary_quality_reader_sha256=h.QUALITY_SHA,
            primary_quality_files=primary_files, primary_quality_transport=primary_transport)
        self.primary_report = report("primary", "PASS", self.primary_lock, self.bound, self.shared["pair"])
        self.bound["primary_quality_report_sha256"] = p.sha(p.canonical(self.primary_report))
        self.inputs = self.make_inputs(6000, 121, "6" * 40)
        replica_producer = p.canonical(dict(schema="english-v6-b27-replication-producer-v1", frozen=True))
        self.raw = dict(evaluator_sha256=b"INVENTED replica evaluator\n",
            producer_manifest_sha256=replica_producer,
            external_proof_binding_sha256=p.canonical(self.bound))
        self.replica_spec = self.spec("replication", self.inputs, p.sha(replica_producer),
            p.sha(self.raw["external_proof_binding_sha256"]))
        self.replica_spec.update(primary_quality_lock=self.primary_lock,
            original_primary_pass_required=True, replication_opened=True)
        self.raw["inputs_sha256"] = p.canonical(self.replica_spec)
        self.cfg.update({key: p.sha(raw) for key, raw in self.raw.items()})
        self.result = dict(schema=h.evaluator.SCHEMA, run_id="123", run_head="1" * 40,
            run_attempt=1, prepared_parent="2" * 40,
            registered_inputs_sha256=self.cfg["inputs_sha256"],
            producer_manifest_sha256=self.cfg["producer_manifest_sha256"],
            implementation_sha256=self.cfg["evaluator_sha256"],
            model_pt_sha256=self.models, assessment_sha256=self.bound["assessment_sha256"],
            source_gate="SOURCE_GATE_PASS", source_shortfall={}, panel_gate=gate,
            replay_allowed=False, consumed_unknown=[], invented_note="雪🌱 raw bytes")
        self.result_raw = (json.dumps(self.result, ensure_ascii=False, indent=1) + "\n").encode()
        replica_originals = {
            "result": archive({"RESULT.json": self.result_raw}),
            "producer": archive({"replication_producer/" + path: self.raw[key]
                                 for key, path in h.HASH_FILES.items()}),
            "consumed": archive({"CONSUMPTION.json": b'{ "invented" : true }\n'}),
        }
        self.outputs = {}
        for i, role in enumerate(h.OUTPUT_ROLES):
            ident = 7000 + i
            self.blobs[ident] = replica_originals[role]
            self.outputs[role] = pin(ident, replica_originals[role])
        self.lock = h.build_lock(self.cfg, self.outputs, self.raw, self.result_raw)
        self.replica_report = report("replication", gate, self.lock, self.bound, self.shared["pair"])
        self.run = dict(id=123, head_sha="1" * 40, run_attempt=1, event="push",
            status="completed", conclusion="success", path=h.evaluator.WORKFLOW,
            head_branch=h.evaluator.BRANCH.removeprefix("refs/heads/"),
            repository=dict(full_name=h.REPO, private=False))
        self.commit = dict(sha="1" * 40, parents=[dict(sha="2" * 40)],
            files=[dict(filename=h.evaluator.WORKFLOW)])
        names = h.quality.artifact_names("replication", 123)
        self.inventory = dict(total_count=3, artifacts=[
            dict(id=self.outputs[role]["id"], name=names[role], expired=False,
                 size_in_bytes=self.outputs[role]["bytes"], digest="sha256:" + self.outputs[role]["sha256"],
                 workflow_run=dict(id=123, head_sha="1" * 40)) for role in h.OUTPUT_ROLES])
        self.binder_mutation = None

    def make_inputs(self, start, run, head):
        pins = copy.deepcopy(self.shared)
        for i, role in enumerate(sorted(h.INPUT_ROLES - set(pins))):
            raw = archive({"INVENTED.txt": (str(start) + " " + role).encode()})
            ident = start + i
            self.blobs[ident] = raw
            pins[role] = pin(ident, raw, run, head)
        return pins

    def spec(self, panel, inputs, producer_sha, binding_sha):
        return dict(panel=panel, schema=h.quality.SCHEMAS[panel], scoring_allowed=True,
            producer_manifest_sha256=producer_sha, external_proof_binding_sha256=binding_sha,
            model_pt_sha256=self.models, assessment_sha256=self.primary_bound["assessment_sha256"],
            inputs={role: inputs[role] for role in ("B0", "B9", "pair", "source", "screen")},
            native_inputs={provider: inputs["native_" + provider] for provider in h.PROVIDERS},
            upstream_producers={role: inputs[role + "_producer"] for role in ("train", "source")})

    def gh(self, endpoint):
        self.requests.append(endpoint)
        self.events.append("GET " + endpoint)
        if endpoint == "actions/runs/123":
            return p.canonical(self.run)
        if endpoint == "commits/" + "1" * 40 + "?per_page=100":
            return p.canonical(self.commit)
        if endpoint == "actions/runs/123/artifacts?per_page=100":
            return p.canonical(self.inventory)
        if endpoint.startswith("actions/artifacts/") and endpoint.endswith("/zip"):
            return self.blobs[int(endpoint.split("/")[2])]
        raise AssertionError("unexpected or forbidden network endpoint: " + endpoint)

    def fetch_files(self, directory, files, transport, reuse=None):
        self.events.append("FETCH " + directory.name)
        if reuse is not None:
            raise AssertionError("no second transport for shared proof")
        body = self.train_bytes if directory.name == "training-origin" else self.primary_bytes
        expected = self.bound["files"] if directory.name == "training-origin" else self.bound["primary_quality_files"]
        locations = self.bound["transport"] if directory.name == "training-origin" else self.bound["primary_quality_transport"]
        if files != expected or transport != locations or set(body) != set(files):
            raise AssertionError("complete original fixture transport")
        directory.mkdir()
        for path, raw in body.items():
            target = directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        if p.inventory(directory) != files:
            raise AssertionError("complete byte inventory")

    def reader(self, directory, lock, proof):
        self.events.append("READER")
        if self.gate != "NO_PROMOTION":
            raise AssertionError("PASS must not preverify replica")
        if directory != proof.parent / "replication-quality" or lock != self.lock:
            raise AssertionError("exact replica reader arguments")
        return copy.deepcopy(self.replica_report)

    def binder(self, **kwargs):
        self.events.append("BINDER")
        if self.gate != "PASS":
            raise AssertionError("NO_PROMOTION cannot enter engineering binder")
        proof, output = kwargs["primary_proof"], kwargs["output"]
        if kwargs["replication_proof"] != proof or kwargs["primary"] != proof / "primary-quality":
            raise AssertionError("shared complete original proof root")
        if kwargs["replication"] != proof.parent / "replication-quality":
            raise AssertionError("replica 18 is outside proof root")
        if p.document(kwargs["primary_lock"].read_bytes()) != self.primary_lock:
            raise AssertionError("actual original PRIMARY lock")
        if p.document(kwargs["replication_lock"].read_bytes()) != self.lock:
            raise AssertionError("actual original replica lock")
        doc = copy.deepcopy(self.inert_doc)
        stem = h.engineering_binder.eng.STEM
        doc.update(enabled=True, quality_reader_sha256=h.QUALITY_SHA,
            model_pt_sha256=self.models, original_training_report_sha256=h.TRAIN_REPORT_SHA,
            candidate_files={stem + "." + ext: dict(bytes=12345, sha256=digest)
                for ext, digest in (("pt", self.models["challenger"]), ("npz", "7" * 64), ("onnx", "8" * 64))})
        doc["quality_locks"] = dict(primary=self.primary_lock, replication=self.lock)
        doc["inputs"]["pair"] = self.shared["pair"]
        raw = {"engineering_inputs.json": p.canonical(doc),
               "PRIMARY_READBACK.json": p.canonical(self.primary_report),
               "REPLICATION_READBACK.json": p.canonical(self.replica_report)}
        if self.binder_mutation:
            self.binder_mutation(raw)
        audit = dict(schema="english-v6-b27-reviewed-engineering-binding-v1", status="READY_FOR_REVIEW_ONLY",
            files={name: byte_pin(body) for name, body in raw.items()},
            actual_registration_changed=False, engineering_executed=False, selection_changed=False,
            new_model_calls=0, new_source_calls=0, compiler_calls=0, packs_created=0)
        output.mkdir()
        for name, body in dict(raw, **{"BINDING.json": p.canonical(audit)}).items():
            (output / name).write_bytes(body)
        return audit

    def mocked(self):
        stack = ExitStack()
        self.network = stack.enter_context(mock.patch.object(h.evaluator, "gh", side_effect=self.gh))
        self.fetch = stack.enter_context(mock.patch.object(h.original_proof, "fetch_files", side_effect=self.fetch_files))
        self.full_reader = stack.enter_context(mock.patch.object(h.quality, "verify_quality_original", side_effect=self.reader))
        self.full_binder = stack.enter_context(mock.patch.object(h.engineering_binder, "prepare_inputs", side_effect=self.binder))
        stack.enter_context(mock.patch.object(h, "check_checkout", return_value=(self.raw, self.inert_doc)))
        stack.enter_context(mock.patch.object(h, "load_bound", return_value=self.bound))
        stack.enter_context(mock.patch.object(h.quality, "validate_lock", side_effect=generic_lock_check))
        stack.enter_context(mock.patch.object(h.evaluator, "BASELINES",
            {role: self.inputs[role] for role in ("B0", "B9")}))
        for target, name in ((h.original_proof, "fetch"), (h.quality, "fetch_quality_original"),
                             (h.profile, "require_proof"), (subprocess, "run"), (subprocess, "check_output")):
            stack.enter_context(mock.patch.object(target, name,
                side_effect=AssertionError("no real network, proof, runtime or compiler in invented tests")))
        return stack


class EntryAndMetadata(unittest.TestCase):
    def test_disabled_refuses_before_env_git_network_mkdir_and_cli_overrides(self):
        inert = dict(schema=h.CONFIG_SCHEMA, enabled=False, **dict.fromkeys(h.CONFIG_PINS))
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "absent"
            cfg = Path(temporary) / "disabled.json"
            cfg.write_bytes(p.canonical(inert))
            with mock.patch.object(h.evaluator, "gh") as gh, \
                 mock.patch.object(h.original_proof, "fetch_files") as fetch, \
                 mock.patch.object(h, "certify_once") as certify, \
                 mock.patch.object(Path, "mkdir") as mkdir, \
                 mock.patch.object(subprocess, "run") as run, \
                 mock.patch.object(h.os, "environ", object()):
                with self.assertRaisesRegex(ValueError, "INERT"):
                    h.prepare(inert, work)
                with mock.patch.object(h, "CONFIG", cfg), mock.patch.object(sys, "argv",
                    ["helper", str(work), "--enabled", "true"]):
                    with self.assertRaisesRegex(ValueError, "INERT"):
                        h.main()
                for operation in (gh, fetch, certify, mkdir, run):
                    operation.assert_not_called()
            self.assertFalse(work.exists())

    def test_exact_configuration_types_and_default_future_pins(self):
        actual = p.document(h.CONFIG.read_bytes())
        if actual["enabled"] is False:
            self.assertTrue(all(actual[key] is None for key in h.CONFIG_PINS))
            with self.assertRaisesRegex(ValueError, "INERT"):
                h.validate_config(actual)
        else:
            h.validate_config(actual)
        self.assertEqual(h.validate_config(config()), config())
        for key, value in (("enabled", 1), ("replication_run_id", True), ("replication_run_id", 0),
                           ("replication_run_head", "z" * 40), ("replication_prepared_parent", "1" * 40),
                           ("inputs_sha256", None)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                h.validate_config(dict(config(), **{key: value}))
        with self.assertRaises(ValueError):
            h.validate_config(dict(config(), override=True))

    def test_public_once_workflow_only_entry_and_exact_two_tokens(self):
        head, parent = "3" * 40, "4" * 40
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=h.REPO, GITHUB_EVENT_NAME="push",
            GITHUB_REF=h.BRANCH, GITHUB_RUN_ATTEMPT="1", GITHUB_SHA=head, PREPARED_PARENT=parent)
        event = dict(before=parent, after=head, ref=h.BRANCH, forced=False, deleted=False,
                     repository=dict(full_name=h.REPO, private=False))
        h.validate_entry(env, event, head, parent, [h.WORKFLOW])
        for key, value in (("GITHUB_RUN_ATTEMPT", "2"), ("GITHUB_EVENT_NAME", "pull_request"),
                           ("GITHUB_REF", h.evaluator.BRANCH), ("PREPARED_PARENT", "5" * 40)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                h.validate_entry(dict(env, **{key: value}), event, head, parent, [h.WORKFLOW])
        for changed in ([], [h.WORKFLOW, "automation/extra.py"]):
            with self.assertRaises(ValueError):
                h.validate_entry(env, event, head, parent, changed)
        for field in ("forced", "deleted"):
            with self.assertRaises(ValueError):
                h.validate_entry(env, dict(event, **{field: True}), head, parent, [h.WORKFLOW])
        before = (h.ROOT / h.WORKFLOW).read_bytes()
        token_count = before.count(h.PARENT_TOKEN.encode())
        if token_count == 2:
            current = before.replace(h.PARENT_TOKEN.encode(), parent.encode())
            h.validate_workflow_activation(before, current, parent)
            with self.assertRaises(ValueError):
                h.validate_workflow_activation(before, current + b"# extra\n", parent)
        else:
            self.assertEqual(token_count, 0)  # The actual once-only activation replaces both.
        fake = (h.PARENT_TOKEN + "\n" + h.PARENT_TOKEN + "\n").encode()
        for old, new in ((fake, fake), (fake + h.PARENT_TOKEN.encode(), fake)):
            with self.assertRaises(ValueError):
                h.validate_workflow_activation(old, new, parent)

    def test_final_public_exact_api_and_three_original_artifacts(self):
        f = Fixture()
        self.assertEqual(h.validate_api(f.cfg, f.run, f.commit, f.inventory), f.outputs)
        for key, value in (("status", "in_progress"), ("conclusion", None), ("conclusion", "failure"),
                           ("run_attempt", 2), ("id", 124), ("head_sha", "0" * 40),
                           ("path", h.WORKFLOW), ("event", "workflow_dispatch")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                h.validate_api(f.cfg, dict(f.run, **{key: value}), f.commit, f.inventory)
        private = copy.deepcopy(f.run); private["repository"]["private"] = True
        with self.assertRaises(ValueError):
            h.validate_run(f.cfg, private)
        cases = []
        bad = copy.deepcopy(f.inventory); bad["total_count"] = 4; cases.append(bad)
        bad = copy.deepcopy(f.inventory); bad["artifacts"].pop(); cases.append(bad)
        for field, value in (("expired", True), ("digest", "sha256:bad"),
                             ("size_in_bytes", 0), ("name", "substituted")):
            bad = copy.deepcopy(f.inventory); bad["artifacts"][0][field] = value; cases.append(bad)
        bad = copy.deepcopy(f.inventory); bad["artifacts"][1]["id"] = bad["artifacts"][0]["id"]; cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):
                h.validate_api(f.cfg, f.run, f.commit, bad)
        bad = copy.deepcopy(f.commit); bad["files"].append(dict(filename="automation/extra.py"))
        with self.assertRaises(ValueError):
            h.validate_api(f.cfg, f.run, bad, f.inventory)

    def test_fixed_engineering_code_hashes_and_unbound_registration(self):
        cfg = config()
        cfg.update({key: p.sha((h.ROOT / path).read_bytes()) for key, path in h.HASH_FILES.items()})
        raw, inert = h.check_checkout(cfg)
        self.assertEqual(set(raw), set(h.HASH_FILES))
        self.assertIs(inert["enabled"], False)
        with self.assertRaises(ValueError):
            h.check_checkout(dict(cfg, evaluator_sha256="0" * 64))
        altered = dict(h.FIXED_FILES)
        altered["automation/bind_b27_engineering.py"] = "0" * 64
        with mock.patch.object(h, "FIXED_FILES", altered), self.assertRaises(ValueError):
            h.check_checkout(cfg)


class ProofAndRouting(unittest.TestCase):
    def test_complete_41_18_links_and_primary_original_registration(self):
        f = Fixture()
        with mock.patch.object(h.quality, "validate_lock", side_effect=generic_lock_check):
            h.validate_binding_links(f.cfg, f.raw, f.bound, f.primary_bound, f.primary_raw)
            cases = []
            bad = copy.deepcopy(f.bound); bad["files"].pop(next(iter(bad["files"]))); cases.append(bad)
            bad = copy.deepcopy(f.bound); bad["primary_quality_files"].pop("api/run.json"); cases.append(bad)
            bad = copy.deepcopy(f.bound); bad["training_report_sha256"] = "0" * 64; cases.append(bad)
            bad = copy.deepcopy(f.bound); bad["primary_quality_lock"]["inputs_sha256"] = "0" * 64; cases.append(bad)
            bad = copy.deepcopy(f.bound); bad["primary_quality_reader_sha256"] = "0" * 64; cases.append(bad)
            for bad in cases:
                changed_raw = dict(f.raw, external_proof_binding_sha256=p.canonical(bad))
                cfg = dict(f.cfg, external_proof_binding_sha256=p.sha(changed_raw["external_proof_binding_sha256"]))
                with self.assertRaises(ValueError):
                    h.validate_binding_links(cfg, changed_raw, bad, f.primary_bound, f.primary_raw)
            bad_primary = dict(f.primary_raw, inputs_sha256=f.primary_raw["inputs_sha256"] + b" ")
            with self.assertRaises(ValueError):
                h.validate_binding_links(f.cfg, f.raw, f.bound, f.primary_bound, bad_primary)

    def test_raw_captured_members_build_exact_lock_without_reencoding_result(self):
        f = Fixture()
        self.assertEqual(h.build_lock(f.cfg, f.outputs, f.raw, f.result_raw), f.lock)
        self.assertNotEqual(f.result_raw, p.canonical(f.result))
        self.assertEqual(f.lock["original_result_sha256"], p.sha(f.result_raw))
        for key in h.HASH_FILES:
            bad = dict(f.raw); bad[key] += b" "
            with self.subTest(key=key), self.assertRaises(ValueError):
                h.build_lock(f.cfg, f.outputs, bad, f.result_raw)
        with self.assertRaises(ValueError):
            h.build_lock(f.cfg, {k: v for k, v in f.outputs.items() if k != "consumed"}, f.raw, f.result_raw)

    def test_raw_preliminary_gate_rejects_stop_unknown_replay_and_substitution(self):
        for gate in ("PASS", "NO_PROMOTION"):
            f = Fixture(gate)
            self.assertEqual(h.preliminary_gate(f.cfg, f.result, f.replica_spec, f.lock, f.bound), gate)
        f = Fixture()
        for field, value in (("panel_gate", "STOP"), ("panel_gate", None), ("panel_gate", "FAILED"),
                             ("source_gate", "SOURCE_GATE_FAIL"), ("source_shortfall", {"missing": 1}),
                             ("replay_allowed", True), ("consumed_unknown", ["one"]),
                             ("run_attempt", 2), ("implementation_sha256", "0" * 64),
                             ("prepared_parent", "0" * 40)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                h.preliminary_gate(f.cfg, dict(f.result, **{field: value}), f.replica_spec, f.lock, f.bound)

    def test_exact_12_inputs_and_18_file_transport(self):
        f = Fixture()
        with mock.patch.object(h.evaluator, "BASELINES", {role: f.inputs[role] for role in ("B0", "B9")}):
            self.assertEqual(h.input_pins(f.replica_spec, f.bound), f.inputs)
            bad = copy.deepcopy(f.replica_spec); bad["upstream_producers"].pop("source")
            with self.assertRaises(ValueError):
                h.input_pins(bad, f.bound)
            bad = copy.deepcopy(f.replica_spec); bad["inputs"]["pair"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                h.input_pins(bad, f.bound)
        api = {name: p.canonical(value) for name, value in (
            ("run", f.run), ("commit", f.commit), ("artifacts", f.inventory))}
        files, transport = h.quality_layout(api, f.outputs, f.inputs)
        self.assertEqual(set(files), h.QUALITY_FILES)
        self.assertEqual(len(files), 18)
        self.assertEqual(base64.b64decode(transport["api/run.json"]["original_api_base64"]), api["run"])
        with self.assertRaises(ValueError):
            h.quality_layout(dict(api, extra=b"{}"), f.outputs, f.inputs)
        bad = copy.deepcopy(f.inputs); bad["source"]["id"] = f.outputs["result"]["id"]
        with self.assertRaises(ValueError):
            h.quality_layout(api, f.outputs, bad)

    def test_fully_verified_report_flags_and_same_actual_candidate(self):
        f = Fixture()
        h.check_report(f.replica_report, "replication", "PASS", f.lock, f.bound)
        cases = [dict(f.replica_report, **{key: value}) for key, value in (
            ("panel_gate", "NO_PROMOTION"), ("original_lock_sha256", "0" * 64),
            ("new_model_calls", True), ("new_source_calls", 1),
            ("candidate_engineering_validation_allowed", False),
            ("replication_open_allowed", True), ("promotion_allowed", True),
            ("all_original_arrays_verified", False))]
        for bad in cases:
            with self.assertRaises(ValueError):
                h.check_report(bad, "replication", "PASS", f.lock, f.bound)

    def test_no_promotion_calls_only_one_top_level_reader(self):
        f = Fixture("NO_PROMOTION")
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            with f.mocked():
                raw, extra = h.certify_once("NO_PROMOTION", work / "replication-quality", f.lock,
                    work / "locks", work / "proof", work / "bound", f.bound, f.inert_doc)
            self.assertEqual(p.document(raw), f.replica_report)
            self.assertEqual(extra, {})
            self.assertEqual(f.full_reader.call_count, 1)
            f.full_binder.assert_not_called()
            f.network.assert_not_called()
            f.fetch.assert_not_called()
            self.assertFalse((work / "bound").exists())

    def test_pass_calls_only_binder_and_preserves_all_four_output_bytes(self):
        f = Fixture()
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            locks = work / "locks"; locks.mkdir()
            (locks / "PRIMARY_LOCK.json").write_bytes(p.canonical(f.primary_lock))
            (locks / "REPLICATION_LOCK.json").write_bytes(p.canonical(f.lock))
            with f.mocked():
                raw, extra = h.certify_once("PASS", work / "replication-quality", f.lock, locks,
                    work / "proof", work / "bound", f.bound, f.inert_doc)
            self.assertEqual(raw, (work / "bound/REPLICATION_READBACK.json").read_bytes())
            for path, name in ((h.ENGINEERING_SPEC, "engineering_inputs.json"),
                               (h.PRIMARY_REPORT_PATH, "PRIMARY_READBACK.json"),
                               (h.BINDING_PATH, "BINDING.json")):
                self.assertEqual(extra[path], (work / "bound" / name).read_bytes())
            self.assertEqual(f.full_binder.call_count, 1)
            f.full_reader.assert_not_called()
            f.network.assert_not_called()
            self.assertEqual(set(p.document(extra[h.BINDING_PATH])["files"]),
                             {"engineering_inputs.json", "PRIMARY_READBACK.json", "REPLICATION_READBACK.json"})

    def test_pass_rejects_wrong_original_primary_report_or_changed_engineering_threshold(self):
        for kind in ("primary", "threshold"):
            f = Fixture()
            def mutation(raw, kind=kind):
                name = "PRIMARY_READBACK.json" if kind == "primary" else "engineering_inputs.json"
                value = p.document(raw[name])
                if kind == "primary":
                    value["unregistered_extra"] = True
                else:
                    value["c_max_abs"] = 0.5
                raw[name] = p.canonical(value)
            f.binder_mutation = mutation
            with tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary); locks = work / "locks"; locks.mkdir()
                (locks / "PRIMARY_LOCK.json").write_bytes(p.canonical(f.primary_lock))
                (locks / "REPLICATION_LOCK.json").write_bytes(p.canonical(f.lock))
                with f.mocked(), self.subTest(kind=kind), self.assertRaises(ValueError):
                    h.certify_once("PASS", work / "replication-quality", f.lock, locks,
                        work / "proof", work / "bound", f.bound, f.inert_doc)
                self.assertEqual(f.full_binder.call_count, 1)
                f.full_reader.assert_not_called()


class Orchestration(unittest.TestCase):
    def exercise(self, gate):
        f = Fixture(gate)
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "work"
            output = io.StringIO()
            with f.mocked(), redirect_stdout(output):
                h.prepare(f.cfg, work)
            self.assertEqual(f.fetch.call_count, 2)
            self.assertEqual([args.args[0].name for args in f.fetch.call_args_list],
                             ["training-origin", "primary-quality"])
            self.assertEqual(p.inventory(work / "proof/training-origin"), f.bound["files"])
            self.assertEqual(p.inventory(work / "proof/primary-quality"), f.bound["primary_quality_files"])
            self.assertEqual(set(p.inventory(work / "replication-quality")), h.QUALITY_FILES)
            self.assertEqual(set(item.name for item in (work / "proof").iterdir()),
                             {"training-origin", "primary-quality"})
            self.assertEqual(f.full_binder.call_count, int(gate == "PASS"))
            self.assertEqual(f.full_reader.call_count, int(gate == "NO_PROMOTION"))
            for value in f.shared.values():
                self.assertNotIn("actions/artifacts/" + str(value["id"]) + "/zip", f.requests)
            archive_requests = [value for value in f.requests if value.startswith("actions/artifacts/")]
            self.assertEqual(len(archive_requests), 11)
            self.assertEqual(len(set(archive_requests)), 11)
            self.assertLess(f.events.index("GET actions/artifacts/7002/zip"),
                            f.events.index("FETCH training-origin"))
            self.assertLess(f.events.index("FETCH primary-quality"),
                            f.events.index("BINDER" if gate == "PASS" else "READER"))
            raw_dir = work / "review/raw"
            self.assertEqual((raw_dir / "RESULT.json").read_bytes(), f.result_raw)
            self.assertEqual({path.name for path in (raw_dir / "api").iterdir()},
                             {"run.json", "commit.json", "artifacts.json"})
            paths = h.COMMON_PATHS + (h.PASS_PATHS if gate == "PASS" else ())
            self.assertEqual(set(p.inventory(work / "review/export")), set(paths))
            receipt = p.document((work / "review/export" / h.RECEIPT_PATH).read_bytes())
            self.assertEqual(receipt["candidate_terminal"], gate == "NO_PROMOTION")
            self.assertEqual(receipt["engineering_inputs_ready_for_review"], gate == "PASS")
            for key in ("engineering_started", "selection_changed", "actual_registration_changed",
                        "scientific_execution_authorized", "promotion_allowed", "product_released"):
                self.assertIs(receipt[key], False)
            for key in ("new_model_calls", "new_source_calls", "new_observer_calls", "new_optimizer_updates",
                        "compiler_calls", "packs_created", "new_marker_consumptions", "remote_mutations"):
                self.assertEqual(receipt[key], 0)
            lines = output.getvalue().splitlines()
            self.assertEqual(sum(line.startswith(h.COMPLETE_PREFIX) for line in lines), 1)
            self.assertTrue(lines[-1].startswith(h.COMPLETE_PREFIX))
            manifest = p.document(lines[-1][len(h.COMPLETE_PREFIX):].encode())
            self.assertEqual(manifest["outcome"], gate)
            self.assertEqual(manifest["file_count"], len(paths))

    def test_invented_no_promotion_full_transport_then_only_reader(self):
        self.exercise("NO_PROMOTION")

    def test_invented_pass_full_transport_then_only_original_binder(self):
        self.exercise("PASS")

    def test_nonfinal_stops_before_work_or_proof_or_certification(self):
        f = Fixture(); f.run["status"] = "in_progress"; f.run["conclusion"] = None
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "absent"
            with f.mocked(), self.assertRaises(ValueError):
                h.prepare(f.cfg, work)
            self.assertFalse(work.exists())
            self.assertEqual(f.requests, ["actions/runs/123"])
            f.fetch.assert_not_called(); f.full_binder.assert_not_called(); f.full_reader.assert_not_called()

    def test_raw_stop_retains_evidence_and_never_fetches_large_proof(self):
        f = Fixture("STOP")
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "work"; output = io.StringIO()
            with f.mocked(), redirect_stdout(output), self.assertRaises(ValueError):
                h.prepare(f.cfg, work)
            self.assertEqual((work / "review/raw/RESULT.json").read_bytes(), f.result_raw)
            self.assertEqual(len(list((work / "replication-quality/originals").iterdir())), 3)
            self.assertFalse((work / "proof").exists())
            self.assertFalse((work / "review/export").exists())
            self.assertNotIn(h.COMPLETE_PREFIX, output.getvalue())
            f.fetch.assert_not_called(); f.full_binder.assert_not_called(); f.full_reader.assert_not_called()

    def test_failed_reader_or_binder_never_falls_back_retries_or_completes(self):
        for gate in ("PASS", "NO_PROMOTION"):
            f = Fixture(gate)
            with tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary) / "work"; output = io.StringIO()
                with f.mocked(), redirect_stdout(output):
                    selected = f.full_binder if gate == "PASS" else f.full_reader
                    selected.side_effect = ValueError("invented certification failure")
                    with self.subTest(gate=gate), self.assertRaisesRegex(ValueError, "invented certification"):
                        h.prepare(f.cfg, work)
                self.assertEqual(selected.call_count, 1)
                other = f.full_reader if gate == "PASS" else f.full_binder
                other.assert_not_called()
                self.assertEqual(p.inventory(work / "proof/training-origin"), f.bound["files"])
                self.assertEqual(p.inventory(work / "proof/primary-quality"), f.bound["primary_quality_files"])
                self.assertEqual((work / "review/raw/RESULT.json").read_bytes(), f.result_raw)
                self.assertFalse((work / "review/export").exists())
                self.assertNotIn(h.COMPLETE_PREFIX, output.getvalue())

    def test_truncated_api_corrupt_zip_and_captured_file_reject_before_large_proof(self):
        for kind in ("inventory", "zip", "captured"):
            f = Fixture()
            if kind == "inventory":
                f.inventory["total_count"] = 4
            elif kind == "zip":
                f.blobs[f.outputs["result"]["id"]] += b"corrupt"
            else:
                # Authentic ZIP receipt containing a member that disagrees with the reviewed config.
                changed = dict(f.raw, evaluator_sha256=b"substituted evaluator\n")
                raw = archive({"replication_producer/" + path: changed[key] for key, path in h.HASH_FILES.items()})
                ident = f.outputs["producer"]["id"]; f.blobs[ident] = raw
                f.outputs["producer"].update(byte_pin(raw))
                item = next(item for item in f.inventory["artifacts"] if item["id"] == ident)
                item.update(size_in_bytes=len(raw), digest="sha256:" + p.sha(raw))
            with tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary) / "work"; output = io.StringIO()
                with f.mocked(), redirect_stdout(output), self.subTest(kind=kind), self.assertRaises(ValueError):
                    h.prepare(f.cfg, work)
                self.assertTrue((work / "review/raw/api/artifacts.json").is_file())
                self.assertFalse((work / "proof").exists())
                self.assertNotIn(h.COMPLETE_PREFIX, output.getvalue())
                f.fetch.assert_not_called(); f.full_binder.assert_not_called(); f.full_reader.assert_not_called()


class ExportContracts(unittest.TestCase):
    def test_five_and_eight_files_multichunk_unicode_original_bytes_roundtrip(self):
        for outcome in ("PASS", "NO_PROMOTION"):
            paths = h.COMMON_PATHS + (h.PASS_PATHS if outcome == "PASS" else ())
            files = {path: ((path + " 雪🌱\n") * (1200 if index == 0 else 2)).encode()
                     for index, path in enumerate(paths)}
            records, manifest = h.make_export(files, outcome)
            self.assertEqual(manifest["outcome"], outcome)
            self.assertEqual(manifest["file_count"], len(paths))
            self.assertEqual(manifest["total_bytes"], sum(map(len, files.values())))
            self.assertGreater(manifest["files"][0]["chunks"], 1)
            self.assertEqual([item["path"] for item in manifest["files"]], list(paths))
            for item in manifest["files"]:
                chunks = [record for record in records if record["path"] == item["path"]]
                self.assertEqual([value["index"] for value in chunks], list(range(item["chunks"])))
                self.assertTrue(all(len(value["base64"]) <= h.CHUNK_CHARS for value in chunks))
                raw = base64.b64decode("".join(value["base64"] for value in chunks), validate=True)
                self.assertEqual(raw, files[item["path"]])
                self.assertEqual(byte_pin(raw), {key: item[key] for key in ("bytes", "sha256")})
                for value in chunks:
                    self.assertEqual({key: value[key] for key in item}, item)

    def test_wrong_outcome_extra_missing_empty_and_eight_mb_boundary(self):
        files = dict.fromkeys(h.COMMON_PATHS, b"{}")
        for bad in (dict(files, extra=b"x"), {k: v for k, v in files.items() if k != h.LOCK_PATH},
                    dict(files, **{h.RESULT_PATH: b""})):
            with self.assertRaises(ValueError):
                h.make_export(bad, "NO_PROMOTION")
        with self.assertRaises(ValueError):
            h.make_export(files, "STOP")
        with self.assertRaises(ValueError):
            h.make_export(dict(files, **dict.fromkeys(h.PASS_PATHS, b"{}")), "NO_PROMOTION")
        exact = dict.fromkeys(h.COMMON_PATHS, b"x")
        exact[h.RESULT_PATH] = b"x" * (h.MAX_EXPORT_BYTES - len(h.COMMON_PATHS) + 1)
        _, manifest = h.make_export(exact, "NO_PROMOTION")
        self.assertEqual(manifest["total_bytes"], h.MAX_EXPORT_BYTES)
        exact[h.RESULT_PATH] += b"x"
        with self.assertRaises(ValueError):
            h.make_export(exact, "NO_PROMOTION")


if __name__ == "__main__":
    unittest.main()

