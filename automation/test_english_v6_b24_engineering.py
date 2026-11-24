"""Pure standard-library boundary fixtures: no pack, compiler, audio or model."""
from __future__ import annotations
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import re
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("b24_engineering_contract", HERE / "english_v6_b24_engineering.py")
eng = importlib.util.module_from_spec(spec); spec.loader.exec_module(eng)


def pin(number):
    return dict(id=number, run_id=number, run_head=("%040x" % number), bytes=100, sha256=("%064x" % number))


def dual_fixture():
    models = {"B0": eng.B0_PT, "B9": eng.B9_PT, "control": eng.CONTROL_PT, "challenger": eng.CANDIDATE_PT}
    doc = dict(model_pt_sha256=models, inputs={"pair": pin(10)}, quality_locks={})
    proofs = {}
    for number, panel in enumerate(("primary", "replication"), 1):
        doc["quality_locks"][panel] = dict(result=pin(number), original_result_sha256=("%064x" % (100 + number)))
        proofs[panel] = dict(schema="english-v6-b24-verified-original-quality-v1", panel=panel,
            panel_gate="PASS", model_pt_sha256=models, assessment_sha256=eng.ASSESSMENT_SHA,
            training_design_sha256=eng.DESIGN_SHA, pair_pin=pin(10),
            all_original_arrays_verified=True, original_source_admission_recomputed=True,
            original_producer_git_verified=True, new_model_calls=0, new_source_calls=0, new_observer_calls=0,
            admission=dict(same_job_matched_control=False, historical_control_retrained=False,
                           model_comparison=eng.MODEL_COMPARISON),
            original_result_sha256=doc["quality_locks"][panel]["original_result_sha256"],
            source_inventory=dict(original_count=120,
                pcm_sha256=["%064x" % (number * 1000 + k) for k in range(120)],
                source_group_id=[panel + "-group-" + str(k) for k in range(36)],
                canonical_people=[panel + "-person-" + str(k) for k in range(36)]))
    return doc, proofs


def logits(labels):
    return b"".join(struct.pack("<4f", *[1.0 if cls == label else -1.0 for cls in range(4)]) for label in labels)


class Contracts(unittest.TestCase):
    def test_historical_engineering_artifact_pins_preserve_every_original_byte(self):
        definition=importlib.util.spec_from_file_location('original_b22_engineering_fixture',HERE/'english_v6_b22_engineering.py')
        original=importlib.util.module_from_spec(definition)
        definition.loader.exec_module(original)
        self.assertEqual(eng.FROZEN_INPUTS,original.FROZEN_INPUTS)
        spec=json.loads(eng.SPEC.read_bytes())
        for role,pin in original.FROZEN_INPUTS.items():
            self.assertEqual(spec['inputs'][role],pin)
        before=json.loads(original.PIPELINE_FILES.read_bytes())
        after=json.loads(eng.PIPELINE_FILES.read_bytes())
        for key in ('files','commit','repository'):
            self.assertEqual(before[key],after[key])

    def test_all_engineering_numerical_loops_keep_original_b22_behavior(self):
        def functions(path):
            return {node.name:node for node in ast.parse(path.read_bytes()).body if isinstance(node,ast.FunctionDef)}
        before=functions(HERE/'english_v6_b22_engineering.py')
        after=functions(Path(eng.__file__))
        for name in ('ctc','compare_logits','validate_events','load_candidate','build_sdk','execute'):
            for node in ast.walk(after[name]):
                if isinstance(node,ast.Constant) and isinstance(node.value,str):
                    node.value=node.value.replace('b24','b22').replace('B24','B22')
            self.assertEqual(ast.dump(before[name],include_attributes=False),
                             ast.dump(after[name],include_attributes=False),name)

    def test_prepared_or_bound_registration_preserves_model_roles_without_execution(self):
        with patch.object(eng, "module", side_effect=AssertionError("runtime import")), \
             patch.object(eng, "gh", side_effect=AssertionError("network")):
            result = eng.source_check()
        self.assertEqual((result["model_forwards"], result["compiled_programs"], result["packs_created"]), (0, 0, 0))
        binding = json.loads(eng.SPEC.read_bytes())
        self.assertEqual(binding["model_pt_sha256"]["control"], eng.CONTROL_PT)
        if binding["enabled"]:
            self.assertEqual(result["status"], "BOUND_REQUIRES_ORIGINAL_DUAL_PASS_READBACK")
            self.assertTrue(eng.valid_hash(binding["model_pt_sha256"]["challenger"]))
            self.assertEqual(set(binding["quality_locks"]), {"primary", "replication"})
        else:
            self.assertEqual(result["status"], "INERT_AWAIT_ORIGINAL_DUAL_QUALITY_PASS")
            self.assertIs(binding["enabled"], False)
            self.assertIsNone(binding["model_pt_sha256"]["challenger"])
            self.assertEqual(binding["quality_locks"], {"primary": None, "replication": None})
            self.assertIsNone(binding["inputs"]["pair"])

    def test_api_preserves_control_and_non_utf8_bytes_without_terminal_output(self):
        raw = b"PK\x03\x04\x1b[31m\xff\x00\x80\r\n"
        with patch.object(eng.subprocess, "run") as run:
            run.return_value.stdout = raw
            self.assertEqual(eng.gh("actions/artifacts/123/zip"), raw)
            run.assert_called_once_with(["gh", "api", "repos/" + eng.REPO + "/actions/artifacts/123/zip",
                                         "--allow-escape-sequences"], stdout=eng.subprocess.PIPE,
                                        stderr=eng.subprocess.PIPE, check=True, timeout=180)

    def test_closed_all_runtime_modes_stop_before_import_network_or_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); closed = root / "closed.json"; closed.write_text('{"enabled": false}')
            with patch.object(eng, "SPEC", closed), patch.object(eng, "module", side_effect=AssertionError("import")), \
                 patch.object(eng, "gh", side_effect=AssertionError("network")), patch.object(eng, "guard", side_effect=AssertionError("guard")):
                for action in (lambda: eng.fetch(root / "inputs", root / "pipeline", "f" * 40),
                               lambda: eng.preflight(root / "inputs", root / "pipeline", "f" * 40),
                               lambda: eng.execute(root / "inputs", root / "pipeline", root / "output", "f" * 40)):
                    with self.assertRaisesRegex(ValueError, "INERT"):
                        action()
                self.assertEqual(sorted(p.name for p in root.iterdir()), ["closed.json"])

    def test_only_exact_single_parent_first_public_workflow_push(self):
        parent, head = "a" * 40, "b" * 40
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=eng.REPO, GITHUB_EVENT_NAME="push",
            GITHUB_REF="refs/heads/" + eng.BRANCH, GITHUB_RUN_ATTEMPT="1", GITHUB_SHA=head, GITHUB_RUN_ID="123")
        event = dict(before=parent, after=head, ref=env["GITHUB_REF"], forced=False, deleted=False,
                     repository=dict(full_name=eng.REPO, private=False))
        eng.validate_event(parent, head, [parent], [eng.WORKFLOW], env, event)
        for change in (dict(prepared=eng.PARENT_TOKEN), dict(parents=[parent, "c" * 40]),
                       dict(changed=[eng.WORKFLOW, "automation/model.py"]),
                       dict(env={**env, "GITHUB_RUN_ATTEMPT": "2"}),
                       dict(event={**event, "forced": True}), dict(event={**event, "before": "c" * 40})):
            values = dict(prepared=parent, head=head, parents=[parent], changed=[eng.WORKFLOW], env=env, event=event)
            values.update(change)
            with self.assertRaises(ValueError):
                eng.validate_event(**values)

    def test_dual_original_identity_and_all120_source_independence(self):
        doc, proofs = dual_fixture()
        self.assertEqual(eng.validate_dual_quality(proofs, doc), proofs)
        changes = (
            lambda x: x["replication"].update(panel_gate="NO_PROMOTION"),
            lambda x: x["replication"].update(all_original_arrays_verified=False),
            lambda x: x["replication"]["admission"].update(same_job_matched_control=True),
            lambda x: x["replication"]["admission"].update(historical_control_retrained=True),
            lambda x: x["replication"].update(schema="english-v6-b21-verified-original-quality-v1"),
            lambda x: x["replication"].update(pair_pin=pin(99)),
            lambda x: x["replication"].update(model_pt_sha256={**x["replication"]["model_pt_sha256"], "challenger": "c" * 64}),
            lambda x: x["replication"].update(original_result_sha256=doc["quality_locks"]["replication"]["result"]["sha256"]),
            lambda x: x["replication"]["source_inventory"]["pcm_sha256"].pop(),
            lambda x: x["replication"]["source_inventory"]["canonical_people"].__setitem__(0, x["primary"]["source_inventory"]["canonical_people"][0]),
            lambda x: x["replication"]["source_inventory"]["source_group_id"].__setitem__(0, x["primary"]["source_inventory"]["source_group_id"][0]),
        )
        for change in changes:
            broken = copy.deepcopy(proofs); change(broken)
            with self.assertRaises(ValueError):
                eng.validate_dual_quality(broken, doc)
        with self.assertRaises((ValueError, KeyError)):
            eng.validate_dual_quality({"primary": {"panel_gate": "PASS"}, "replication": {"panel_gate": "PASS"}}, doc)

    def test_matching_dual_summaries_for_another_candidate_still_reject(self):
        doc,proofs=dual_fixture()
        for models in [doc['model_pt_sha256'],*(value['model_pt_sha256'] for value in proofs.values())]:
            models['challenger']='e'*64
        with self.assertRaisesRegex(ValueError,'actual frozen B24'):
            eng.validate_dual_quality(proofs,doc)
        with patch.object(eng,'original_zip',side_effect=AssertionError('no model archive access')):
            with self.assertRaisesRegex(ValueError,'actual completed B24'):
                eng.original_pair(Path('/unopened'),doc)

    def test_shared_original_reader_called_twice_and_failure_never_falls_back(self):
        doc, proofs = dual_fixture(); calls = []
        for panel in proofs:
            doc["quality_locks"][panel]["panel"] = panel
        def verify(directory, lock):
            calls.append(lock["panel"])
            return proofs[lock["panel"]]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "reader.py"; path.write_bytes(b"invented original-reader fixture")
            doc["quality_reader_sha256"] = eng.sha(path.read_bytes())
            reader = types.SimpleNamespace(__file__=str(path), verify_quality_original=verify)
            with patch.object(eng, "module", return_value=reader):
                eng.quality(Path("unused-fixture-directory"), doc)
            self.assertEqual(calls, ["primary", "replication"])
            reader.verify_quality_original = lambda *args: (_ for _ in ()).throw(ValueError("original array missing"))
            with patch.object(eng, "module", return_value=reader), self.assertRaisesRegex(ValueError, "original array missing"):
                eng.quality(Path("unused-fixture-directory"), doc)

    def test_cached_different_reader_rejected_before_quality_fetch_or_readback(self):
        doc, proofs = dual_fixture()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "other_reader.py"; path.write_bytes(b"invented stale cached reader")
            doc["quality_reader_sha256"] = eng.sha(b"invented reviewed reader")
            reader = types.SimpleNamespace(__file__=str(path),
                fetch_quality_original=lambda *args: self.fail("must not fetch through wrong reader"),
                verify_quality_original=lambda *args: self.fail("must not read through wrong reader"))
            with patch.object(eng, "module", return_value=reader), self.assertRaisesRegex(ValueError, "executing original-quality reader"):
                eng.quality(Path("unused-fixture-directory"), doc, fetch=True)

    def test_first_failed_primary_never_fetches_or_opens_replication(self):
        doc, proofs = dual_fixture()
        for panel in proofs:
            doc["quality_locks"][panel]["panel"] = panel
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "reader.py"; path.write_bytes(b"invented original reader fixture")
            doc["quality_reader_sha256"] = eng.sha(path.read_bytes())
            reader = types.SimpleNamespace(__file__=str(path),
                fetch_quality_original=lambda folder, lock: calls.append(("fetch", lock["panel"])),
                verify_quality_original=lambda folder, lock: (calls.append(("verify", lock["panel"]))
                    or {**proofs[lock["panel"]], "panel_gate": "NO_PROMOTION"}))
            with patch.object(eng, "module", return_value=reader), self.assertRaises(ValueError):
                eng.quality(Path(temp), doc, fetch=True)
        self.assertEqual(calls, [("fetch", "primary"), ("verify", "primary")])

    def test_bound_metadata_without_full_dual_pass_never_fetches_candidate_or_engineering_pcm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(eng, "protocol", return_value={"enabled": True}), \
                 patch.object(eng, "guard", return_value={"run_id": 123}), \
                 patch.object(eng, "one_execution"), patch.object(eng, "producer"), \
                 patch.object(eng, "quality", side_effect=ValueError("original PRIMARY arrays missing")), \
                 patch.object(eng, "authenticate_remote", side_effect=AssertionError("candidate or PCM fetch")), \
                 patch.object(eng, "original_pair", side_effect=AssertionError("candidate access")), \
                 patch.object(eng, "engineering_sources", side_effect=AssertionError("engineering PCM access")):
                with self.assertRaisesRegex(ValueError, "original PRIMARY arrays missing"):
                    eng.fetch(root / "inputs", root / "pipeline", "a" * 40)
            self.assertEqual(sorted(p.name for p in (root / "inputs").iterdir()), ["quality"])

    def test_original_b24_candidate_and_historical_b23_export_paths(self):
        payloads = {arm: {suffix: (arm + "/" + suffix).encode() * 2000 for suffix in ("pt", "npz", "onnx")}
                    for arm in ("control", "challenger")}
        receipt = dict(design_sha256=eng.DESIGN_SHA, assessment_sha256=eng.ASSESSMENT_SHA,
                       same_job_matched_control=False, new_control_optimizer_updates=0)
        for arm in payloads:
            receipt[arm] = dict(files={Path(eng.STEMS[arm]).name + "." + suffix:
                dict(bytes=len(raw), sha256=eng.sha(raw)) for suffix, raw in payloads[arm].items()})
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w") as archive:
            archive.writestr("RESULT.json", json.dumps(receipt))
            for arm, contents in payloads.items():
                for suffix, data in contents.items():
                    archive.writestr(eng.STEMS[arm] + "." + suffix, data)
        doc = dict(inputs={"pair": {**pin(1), "bytes": len(raw.getvalue()), "sha256": eng.sha(raw.getvalue())}},
                   candidate_files=receipt["challenger"]["files"],
                   model_pt_sha256={arm: eng.sha(values["pt"]) for arm, values in payloads.items()})
        calls = []
        primary = types.SimpleNamespace(STEMS=eng.STEMS, MODEL_COMPARISON=eng.MODEL_COMPARISON,
            CONTROL_PT=eng.CONTROL_PT, validate_pair_receipt=lambda value, spec: calls.append((value, spec)))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "pair.zip").write_bytes(raw.getvalue())
            with patch.object(eng, "module", return_value=primary) as importer, \
                 patch.object(eng, "CANDIDATE_PT", eng.sha(payloads['challenger']['pt'])):
                self.assertEqual(eng.original_pair(root, doc), payloads["challenger"])
                importer.assert_called_once_with("english_v6_b24_primary_eval")
                self.assertEqual(calls, [(receipt, doc)])
                bad = copy.deepcopy(doc); bad["model_pt_sha256"]["control"] = "0" * 64
                with self.assertRaises(ValueError): eng.original_pair(root, bad)
        self.assertEqual(eng.STEMS["control"], "control/models/mero43-b23-challenger192")
        self.assertEqual(eng.STEMS["challenger"], "challenger/models/mero43-b24-challenger192")

    def test_full_producer_inherits_dynamic_nonpython_and_hidden_resources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); folder = root / "automation"; folder.mkdir()
            original = dict(schema='english-v6-b24-complete-git-producer-v1', original_b23_dependency_files=306,
                files={"kws_data_contract/identity.py": {}, ".github/triggers/old.json": {},
                       "research/initial/LOG.json": {},
                       **{'research/frozen/dependency-' + str(i) + '.json': {} for i in range(335)}})
            (folder / "english_v6_b24_source_training_producer.json").write_text(json.dumps(original))
            expected = set(original["files"])
            for panel in ("primary", "replication"):
                value = dict(frozen=True, resources={"research/" + panel + ".json": "a" * 64},
                    source=dict(files={"automation/dynamic.py": "b" * 64}, resources={".github/triggers/dynamic.json": "c" * 64}),
                    upstream_manifests={"train": {"files": {"research/retained/source.json": {}}}})
                value[panel] = dict(files={"automation/literal.py": "d" * 64})
                (folder / ("english_v6_b24_" + panel + "_producer.json")).write_text(json.dumps(value))
                expected |= set(value["resources"]) | set(value["source"]["files"]) | set(value["source"]["resources"])
                expected |= set(value[panel]["files"]) | {"research/retained/source.json"}
            required = {eng.WORKFLOW, "automation/english_v6_b24_engineering.py", "automation/test_english_v6_b24_engineering.py",
                "automation/" + eng.SPEC.name, "automation/" + eng.TEMPLATE.name, "automation/" + eng.PIPELINE_FILES.name,
                "automation/" + eng.QUALITY_READER + ".py"}
            tracked = expected | required | {str(path.relative_to(root)) for path in folder.iterdir()} | {"automation/unseen_dynamic.py"}
            selected = eng.producer_paths(tracked, root)
            self.assertTrue(expected | required <= selected)
            self.assertIn("automation/unseen_dynamic.py", selected)
            with self.assertRaises(ValueError):
                eng.producer_paths(tracked - {"research/retained/source.json"}, root)

    def test_full_frame_argmax_ctc_and_numeric_thresholds(self):
        labels = [0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 1, 2, 3]
        raw = logits(labels)
        result = eng.compare_logits(raw, raw, .001, .0001)
        self.assertEqual(result["event_frames"], [7, 12])
        self.assertEqual(result["event_samples"], [1632, 2432])
        self.assertEqual(result["tokens"], [1, 2, 3, 1, 2, 3])
        corrupted = bytearray(raw); struct.pack_into("<f", corrupted, 0, float("nan"))
        changed = logits([1, *labels[1:]])
        excessive = bytearray(raw); struct.pack_into("<f", excessive, 0, 1.01)
        for invalid in (raw[:-16], bytes(corrupted), changed, bytes(excessive)):
            with self.assertRaises(ValueError):
                eng.compare_logits(raw, invalid, .001, .0001)
        tie_reference = struct.pack("<4f", 1, 1, 0, 0)
        tie_changed = struct.pack("<4f", 1, 1.00001, 0, 0)
        with self.assertRaisesRegex(ValueError, "token parity"):
            eng.compare_logits(tie_reference, tie_changed, .001, .0001)

    def test_csv_requires_event_frames_and_all_sample_clocks_not_counts_only(self):
        expected = eng.ctc([0, 1, 0, 2, 0, 3])
        original = "".join(str(i) + "," + str(512 + 160 * i) + "," + str(int(i == 5)) + "\n" for i in range(6)).encode()
        eng.validate_events(original, 6, expected)
        wrong_frame = original.replace(b"4,1152,0", b"4,1152,1").replace(b"5,1312,1", b"5,1312,0")
        wrong_clock = original.replace(b"5,1312,1", b"5,1313,1")
        for changed in (wrong_frame, wrong_clock, original.rsplit(b"\n", 2)[0] + b"\n"):
            with self.assertRaises(ValueError):
                eng.validate_events(changed, 6, expected)

    def test_consumed_unknown_and140_entry_ceiling_precede_any_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = eng.Ledger(Path(temp))
            ledger.begin("case0", "pytorch")
            self.assertEqual(json.loads((Path(temp) / "case0.json").read_bytes())["status"], "consumed_unknown")
            with self.assertRaises(ValueError):
                ledger.begin("case0", "pytorch")
            ledger.complete("case0")
            with self.assertRaises(ValueError):
                ledger.complete("case0")
            for i in range(1, eng.MAX_CALLS):
                ledger.begin("case" + str(i), "pytorch")
            with self.assertRaises(ValueError):
                ledger.begin("extra", "pytorch")
            self.assertEqual(len(ledger.entered), 140)

    def test_archives_reject_duplicate_traversal_symlink_and_wrong_original_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.zip"
            for kind in ("valid", "traversal", "symlink", "duplicate"):
                raw = io.BytesIO()
                with zipfile.ZipFile(raw, "w") as archive:
                    name = "../out" if kind == "traversal" else "receipt.txt"
                    info = zipfile.ZipInfo(name)
                    if kind == "symlink":
                        info.create_system = 3; info.external_attr = 0o120777 << 16
                    archive.writestr(info, b"invented fixture")
                    if kind == "duplicate":
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore"); archive.writestr(name, b"second")
                payload = raw.getvalue(); path.write_bytes(payload)
                item = {**pin(1), "bytes": len(payload), "sha256": eng.sha(payload)}
                if kind == "valid":
                    with eng.original_zip(path, item) as archive:
                        self.assertEqual(archive.read("receipt.txt"), b"invented fixture")
                    with self.assertRaises(ValueError):
                        eng.original_zip(path, {**item, "sha256": "0" * 64})
                else:
                    with self.assertRaises(ValueError):
                        eng.original_zip(path, item)

    def test_workflow_parent_binding_and_source_identity_capture_preserve_hidden_files(self):
        workflow = (HERE.parent / eng.WORKFLOW).read_bytes(); template = eng.TEMPLATE.read_bytes()
        text = workflow.decode()
        parent = re.search(r"PREPARED_PARENT: '([^']+)'", text).group(1)
        self.assertTrue(parent == eng.PARENT_TOKEN or eng.valid_hash(parent, 40))
        self.assertEqual(workflow, template.replace(eng.PARENT_TOKEN.encode(), parent.encode()))
        self.assertIn(eng.PARENT_TOKEN.encode(), template)
        self.assertNotIn("workflow_dispatch", text)
        self.assertEqual(text.count("include-hidden-files: true"), text.count("uses: actions/upload-artifact@"))
        self.assertIn(eng.PIPELINE_COMMIT, text)
        pins = json.loads(eng.PIPELINE_FILES.read_bytes())
        self.assertEqual(pins["commit"], eng.PIPELINE_COMMIT)
        self.assertEqual(len(pins["files"]), 13)
        self.assertIn("research/english_v5_mero/pack_mero43.py", pins["files"])
        self.assertTrue(all(eng.valid_hash(p["git_blob_sha1"], 40) and p["bytes"] > 0 for p in pins["files"].values()))


if __name__ == "__main__":
    unittest.main()
