"""Pure standard-library boundary fixtures: no pack, compiler, audio or model."""
from __future__ import annotations
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
spec = importlib.util.spec_from_file_location("b21_engineering_contract", HERE / "english_v6_b21_engineering.py")
eng = importlib.util.module_from_spec(spec); spec.loader.exec_module(eng)


def pin(number):
    return dict(id=number, run_id=number, run_head=("%040x" % number), bytes=100, sha256=("%064x" % number))


def dual_fixture():
    models = {"B0": eng.B0_PT, "B9": eng.B9_PT, "control": "a" * 64, "challenger": "b" * 64}
    doc = dict(model_pt_sha256=models, inputs={"pair": pin(10)}, quality_locks={})
    proofs = {}
    for number, panel in enumerate(("primary", "replication"), 1):
        doc["quality_locks"][panel] = dict(result=pin(number), original_result_sha256=("%064x" % (100 + number)))
        proofs[panel] = dict(panel_gate="PASS", model_pt_sha256=models, assessment_sha256=eng.ASSESSMENT_SHA,
            training_design_sha256=eng.DESIGN_SHA, pair_pin=pin(10),
            original_result_sha256=doc["quality_locks"][panel]["original_result_sha256"],
            source_inventory=dict(original_count=120,
                pcm_sha256=["%064x" % (number * 1000 + k) for k in range(120)],
                source_group_id=[panel + "-group-" + str(k) for k in range(36)],
                canonical_people=[panel + "-person-" + str(k) for k in range(36)]))
    return doc, proofs


def logits(labels):
    return b"".join(struct.pack("<4f", *[1.0 if cls == label else -1.0 for cls in range(4)]) for label in labels)


class Contracts(unittest.TestCase):
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
