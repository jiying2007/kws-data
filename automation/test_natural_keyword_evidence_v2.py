#!/usr/bin/env python3
"""Invented transcript/receipt tests only; no models or acoustic assertions."""
from __future__ import annotations
import ast
from copy import deepcopy
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("keyword_v2_under_test", HERE / "natural_keyword_evidence_v2.py")
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def pin(text):
    return hashlib.sha256(text.encode()).hexdigest()


def decision(targets=None):
    known = targets is not None
    return {"full_text_quorum": known, "ctc_mask": known,
            "ctc_weight": 0.5 if known else 0.0,
            "keyword_targets": list(targets) if known else [None, None],
            "keyword_mask": [known, known], "full_text": "保留原全文" if known else None,
            "selected_raw_text": "保留原全文。" if known else None, "event_times": None}


def observations(texts=None):
    texts = texts or ["今天你好小窝呀", "你好小窝天气不错", "我说你好小窝"]
    return [{"model": model, "status": "complete", "requested": True,
             "usable": True, "decoder_terminated": True, "quality_flags": [],
             "normalized_text": text, "text": text, "native_receipt_sha256": pin(model)}
            for model, text in zip(v.MODELS, texts)]


def row(name="r1", texts=None, targets=None):
    return {"source_id": name, "split": "development", "source_family_id": "family-" + name,
            "dataset_id": "invented-dataset", "session_id": "session-" + name,
            "parent_pcm_sha256": pin("parent-" + name),
            "window_pcm_sha256": pin("window-" + name),
            "window_wav_sha256": pin("wav-" + name),
            "start_sample": 0, "end_sample": 16000, "sample_rate_hz": 16000,
            "observations": observations(texts), "decision": decision(targets)}


def upstream(rows=None):
    return {"schema": v.UPSTREAM_SCHEMA, "rows": rows or [row()],
            "boundary": {"training_admitted": False, "independent_test": False,
                         "natural_far_available": False}}


def report(rows=None):
    # Identity normalization is sufficient for these already-normalized fixtures.
    # Production imports the unchanged canonical normalizer from the verified tree.
    return v.project_report(upstream(rows), pin("invented-upstream-manifest"), normalize=lambda t: t)


class ProjectionTests(unittest.TestCase):
    def test_positive_without_fulltext(self):
        actual = v.project_keywords(observations(), decision())
        self.assertEqual(actual[0]["machine_candidate_state"], "present")
        self.assertEqual(actual[0]["candidate_target"], 1)
        self.assertTrue(actual[0]["candidate_mask"])
        self.assertFalse(actual[0]["training_mask"])
        self.assertIsNone(actual[0]["loss_weight"])
        self.assertIsNone(actual[0]["event_times"])
        self.assertEqual(actual[1]["machine_candidate_state"], "unknown")

    def test_three_absences_stay_unknown(self):
        actual = v.project_keywords(observations(["今天晴天", "明天下雨", "正在吃饭"]), decision())
        self.assertTrue(all(r["candidate_target"] is None and not r["candidate_mask"] for r in actual))
        self.assertEqual(actual[0]["reason"], "three_non_detections_do_not_prove_absence")

    def test_fulltext_absence_is_inherited_not_promoted(self):
        actual = v.project_keywords(observations(["天气好"] * 3), decision([0, 0]))
        self.assertEqual([r["machine_candidate_state"] for r in actual], ["absent", "absent"])
        self.assertTrue(all(not r["training_mask"] and r["source"] == "inherited" for r in actual))

    def test_fulltext_positive_and_negative_inherited(self):
        actual = v.project_keywords(observations(), decision([1, 0]))
        self.assertEqual([r["candidate_target"] for r in actual], [1, 0])

    def test_per_keyword_independent(self):
        actual = v.project_keywords(observations(["你好小窝小窝小窝", "你好小窝", "你好小窝"]), decision())
        self.assertTrue(actual[0]["candidate_mask"])
        self.assertFalse(actual[1]["candidate_mask"])

    def test_both_keywords_can_be_present(self):
        actual = v.project_keywords(observations(["你好小窝和小窝小窝"] * 3), decision())
        self.assertEqual([r["candidate_target"] for r in actual], [1, 1])

    def test_two_votes_do_not_pass(self):
        actual = v.project_keywords(observations(["你好小窝", "你好小窝", "你好小屋"]), decision())
        self.assertFalse(actual[0]["candidate_mask"])

    def test_homophones_not_rewritten(self):
        actual = v.project_keywords(observations(["你好小屋"] * 3), decision())
        self.assertIsNone(actual[0]["candidate_target"])

    def test_noncontiguous_not_promoted(self):
        actual = v.project_keywords(observations(["你好这个小窝"] * 3), decision())
        self.assertFalse(actual[0]["candidate_mask"])

    def test_all_vote_patterns(self):
        for bits in itertools.product((False, True), repeat=3):
            with self.subTest(bits=bits):
                texts = ["你好小窝" if b else "普通说话" for b in bits]
                actual = v.project_keywords(observations(texts), decision())[0]
                self.assertEqual(actual["candidate_mask"], all(bits))
                self.assertEqual(actual["candidate_target"], 1 if all(bits) else None)

    def test_unusable_decode(self):
        values = observations(); values[2]["usable"] = False
        self.assertTrue(all(not r["candidate_mask"] for r in v.project_keywords(values, decision())))

    def test_nonterminated_decode(self):
        values = observations(); values[2]["decoder_terminated"] = False
        self.assertFalse(v.project_keywords(values, decision())[0]["candidate_mask"])

    def test_quality_flags(self):
        values = observations(); values[1]["quality_flags"] = ["incomplete"]
        self.assertFalse(v.project_keywords(values, decision())[0]["candidate_mask"])

    def test_empty_decode(self):
        values = observations(); values[2]["normalized_text"] = ""
        self.assertFalse(v.project_keywords(values, decision())[0]["candidate_mask"])

    def test_missing_model_rejected(self):
        with self.assertRaises(ValueError): v.project_keywords(observations()[:2], decision())

    def test_duplicate_model_rejected(self):
        values = observations(); values[2]["model"] = "qwen06"
        with self.assertRaises(ValueError): v.project_keywords(values, decision())

    def test_reordered_model_rejected(self):
        with self.assertRaises(ValueError): v.project_keywords(list(reversed(observations())), decision())

    def test_not_requested_rejected(self):
        values = observations(); values[2]["requested"] = False
        with self.assertRaises(ValueError): v.project_keywords(values, decision())

    def test_incomplete_result_rejected(self):
        values = observations(); values[2]["status"] = "unknown"
        with self.assertRaises(ValueError): v.project_keywords(values, decision())

    def test_truthy_nonboolean_rejected(self):
        for field in ("usable", "decoder_terminated"):
            for bad in (1, "true", None):
                values = observations(); values[2][field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    v.project_keywords(values, decision())

    def test_false_mask_with_zero_target_rejected(self):
        value = decision(); value["keyword_targets"] = [0, 0]
        with self.assertRaises(ValueError): v.project_keywords(observations(), value)

    def test_bool_target_rejected(self):
        value = decision([1, 0]); value["keyword_targets"][0] = True
        with self.assertRaises(ValueError): v.project_keywords(observations(), value)

    def test_mask_quorum_conflict(self):
        value = decision(); value["keyword_mask"] = [True, False]; value["keyword_targets"] = [1, None]
        with self.assertRaises(ValueError): v.project_keywords(observations(), value)

    def test_ctc_quorum_conflict(self):
        value = decision(); value["ctc_mask"] = True
        with self.assertRaises(ValueError): v.project_keywords(observations(), value)

    def test_inherited_target_conflict(self):
        with self.assertRaises(ValueError): v.project_keywords(observations(), decision([0, 0]))

    def test_dirty_inherited_quorum(self):
        values = observations(); values[1]["usable"] = False
        with self.assertRaises(ValueError): v.project_keywords(values, decision([1, 0]))

    def test_no_input_mutation(self):
        values, old = observations(), decision()
        expected = deepcopy((values, old))
        v.project_keywords(values, old)
        self.assertEqual((values, old), expected)


class ReportTests(unittest.TestCase):
    def test_preserve_fulltext_and_role(self):
        source = row(targets=[1, 0]); expected = deepcopy(source)
        result = report([source])
        self.assertEqual(source, expected)
        self.assertEqual(result["rows"][0]["original_fulltext_decision"], source["decision"])
        self.assertEqual(result["rows"][0]["split"], "development")
        self.assertEqual(result["boundary"], v.BOUNDARY)

    def test_report_independent_of_source_aliasing(self):
        source = row(targets=[1, 0]); result = report([source])
        result["rows"][0]["original_fulltext_decision"]["keyword_mask"][0] = False
        self.assertTrue(source["decision"]["keyword_mask"][0])

    def test_zero_negative_exposure(self):
        result = report([row(texts=["天气好"] * 3, targets=[0, 0])])
        self.assertEqual(result["boundary"]["accepted_negative_seconds"], 0)
        self.assertFalse(result["boundary"]["natural_far_available"])

    def test_no_duplicate_source(self):
        with self.assertRaises(ValueError): report([row(), row()])

    def test_family_leak_rejected(self):
        a, b = row("a"), row("b"); b["source_family_id"] = a["source_family_id"]; b["split"] = "train"
        with self.assertRaises(ValueError): report([a, b])

    def test_pcm_leak_rejected(self):
        a, b = row("a"), row("b"); b["window_pcm_sha256"] = a["parent_pcm_sha256"]; b["split"] = "train"
        with self.assertRaises(ValueError): report([a, b])

    def test_parent_metadata_conflict(self):
        a, b = row("a"), row("b"); b["parent_pcm_sha256"] = a["parent_pcm_sha256"]
        with self.assertRaises(ValueError): report([a, b])

    def test_duplicate_interval_rejected(self):
        a, b = row("a"), row("b")
        for key in ("parent_pcm_sha256", "source_family_id", "session_id"):
            b[key] = a[key]
        with self.assertRaises(ValueError): report([a, b])

    def test_bad_sha_rejected(self):
        value = row(); value["window_pcm_sha256"] = "claimed-sha"
        with self.assertRaises(ValueError): report([value])

    def test_bad_role_rejected(self):
        value = row(); value["split"] = "test"
        with self.assertRaises(ValueError): report([value])

    def test_bad_interval_rejected(self):
        for field, value in (("start_sample", True), ("end_sample", 0),
                             ("sample_rate_hz", 24000), ("end_sample", 480001)):
            source = row(); source[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): report([source])

    def test_row_bound_rejected(self):
        value = upstream(); value["rows"] = []
        with self.assertRaises(ValueError): v.project_report(value, pin("x"), normalize=lambda t:t)
        value["rows"] = [row()] * 1025
        with self.assertRaises(ValueError): v.project_report(value, pin("x"), normalize=lambda t:t)

    def test_wrong_schema_rejected(self):
        value = upstream(); value["schema"] = "some-freeform-json"
        with self.assertRaises(ValueError): v.project_report(value, pin("x"), normalize=lambda t:t)

    def test_scope_not_promoted(self):
        for field in ("training_admitted", "independent_test", "natural_far_available"):
            value = upstream(); value["boundary"][field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                v.project_report(value, pin("x"), normalize=lambda t:t)

    def test_wrong_upstream_manifest_rejected(self):
        with self.assertRaises(ValueError): v.project_report(upstream(), "no-pin", normalize=lambda t:t)

    def test_production_normalizer_is_used(self):
        calls = []
        def normalize(text):
            calls.append(text); return text
        v.project_report(upstream(), pin("x"), normalize=normalize)
        self.assertEqual(len(calls), 3)

    def test_audit_blind_no_transcripts_or_predictions(self):
        blind, owner = v.make_audit_plan(report(), salt="frozen-audit-salt-001", limit=32)
        self.assertEqual(blind["status"], "planned_not_annotated")
        self.assertNotIn("你好小窝", v.canonical(blind).decode())
        self.assertNotIn("source_id", v.canonical(blind).decode())
        for entry in blind["rows"]:
            self.assertEqual(set(entry), {"blind_id", "audio_wav_sha256", "sample_rate_hz", "sample_count"})
        self.assertFalse(owner["prevalence_estimate_allowed"])

    def test_audit_deterministic_and_bounded(self):
        r = report([row(str(i)) for i in range(10)])
        a = v.make_audit_plan(r, salt="frozen-audit-salt-001", limit=3)
        b = v.make_audit_plan(r, salt="frozen-audit-salt-001", limit=3)
        self.assertEqual(a, b)
        self.assertEqual(a[0]["selected_rows"], 3)
        self.assertEqual(len({x["blind_id"] for x in a[0]["rows"]}), 3)

    def test_audit_covers_strata(self):
        r = report([row("present"), row("unknown", texts=["a", "b", "c"]),
                    row("absent", texts=["hello"]*3, targets=[0,0])])
        _, owner = v.make_audit_plan(r, salt="frozen-audit-salt-001", limit=3)
        self.assertEqual(len({x["stratum"] for x in owner["rows"]}), 3)

    def test_audit_salt_and_limits(self):
        for salt, limit in (("short", 1), ("x"*16, True), ("x"*16, 0), ("x"*16, 129)):
            with self.subTest(salt=salt, limit=limit), self.assertRaises(ValueError):
                v.make_audit_plan(report(), salt=salt, limit=limit)


class PackageTests(unittest.TestCase):
    """Mocked upstream replay tests: do NOT establish real native evidence replay."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.upstream = self.root / "upstream"; self.upstream.mkdir()
        self.output = self.root / "output"
        self.report = report()
        self.mock = patch.object(v, "_replay", return_value=deepcopy(self.report))
        self.replay = self.mock.start()
        self.addCleanup(self.mock.stop)
        self.addCleanup(self.temp.cleanup)

    def create(self):
        return v.create(self.upstream, pin("upstream"), self.output,
                        salt="frozen-audit-salt-001", limit=32)

    def verify(self):
        return v.verify(self.upstream, pin("upstream"), self.output,
                        v.digest((self.output / "MANIFEST.json").read_bytes()))

    def rewrite(self, name, mutator):
        path = self.output / name
        value = json.loads(path.read_bytes()); mutator(value)
        path.write_bytes(v.canonical(value) + b"\n")
        mp = self.output / "MANIFEST.json"; manifest = json.loads(mp.read_bytes())
        manifest["files"][name] = v.record(path.read_bytes())
        mp.write_bytes(v.canonical(manifest) + b"\n")

    def test_create_verify_and_replay_count(self):
        self.assertEqual(self.create()["status"], "created")
        self.assertEqual(self.verify()["status"], "verified")
        self.assertEqual(self.replay.call_count, 2)
        self.assertFalse(self.verify()["boundary"]["training_admitted"])

    def test_does_not_change_upstream(self):
        marker = self.upstream / "marker"; marker.write_bytes(b"retained bytes")
        stat0 = marker.stat(); self.create(); self.verify()
        self.assertEqual(marker.read_bytes(), b"retained bytes")
        self.assertEqual(marker.stat().st_mtime_ns, stat0.st_mtime_ns)

    def test_private_permissions(self):
        self.create()
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o700)
        self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in self.output.iterdir()))

    def test_no_overwrite(self):
        self.create()
        before = (self.output / "MANIFEST.json").read_bytes()
        with self.assertRaises(ValueError): self.create()
        self.assertEqual((self.output / "MANIFEST.json").read_bytes(), before)

    def test_missing_seal_failure_before_output(self):
        self.replay.side_effect = ValueError("missing upstream native seal")
        with self.assertRaises(ValueError): self.create()
        self.assertFalse(self.output.exists())

    def test_wrong_outer_manifest(self):
        self.create()
        with self.assertRaises(ValueError): v.verify(self.upstream, pin("upstream"), self.output, "0"*64)

    def test_rehashing_report_does_not_forge_result(self):
        self.create()
        self.rewrite("report.json", lambda x:x["boundary"].update(training_admitted=True))
        with self.assertRaises(ValueError): self.verify()

    def test_rehashing_blind_payload_does_not_forge_selection(self):
        self.create()
        self.rewrite("audit-blind.json", lambda x:x.update(human_gold=True))
        with self.assertRaises(ValueError): self.verify()

    def test_source_pin_changed(self):
        self.create()
        mp = self.output / "MANIFEST.json"; m = json.loads(mp.read_bytes()); m["producer"]["sha256"] = "0"*64
        mp.write_bytes(v.canonical(m))
        with self.assertRaises(ValueError): self.verify()

    def test_extra_file_rejected(self):
        self.create(); (self.output / "extra.json").write_bytes(b"{}")
        with self.assertRaises(ValueError): self.verify()

    def test_extra_directory_rejected(self):
        self.create(); (self.output / "extra").mkdir()
        with self.assertRaises(ValueError): self.verify()

    def test_hardlink_rejected(self):
        self.create(); os.link(self.output / "report.json", self.root / "alias")
        with self.assertRaises(ValueError): self.verify()

    def test_symlink_payload_rejected(self):
        self.create(); p = self.output / "report.json"; raw = p.read_bytes(); p.unlink()
        other = self.root / "other.json"; other.write_bytes(raw); p.symlink_to(other)
        with self.assertRaises(ValueError): self.verify()

    def test_symlink_parent_rejected(self):
        alias = self.root / "alias"; alias.symlink_to(self.upstream, target_is_directory=True)
        with self.assertRaises(ValueError): v.real_path(alias / "x")

    def test_traversal_rejected(self):
        with self.assertRaises(ValueError): v.real_path(self.root / ".." / "outside")

    def test_output_inside_input_rejected(self):
        with self.assertRaises(ValueError):
            v.create(self.upstream, pin("upstream"), self.upstream / "output", salt="x"*16, limit=2)

    def test_relocated_package_replays(self):
        self.create(); moved = self.root / "relocated"; self.output.rename(moved); self.output = moved
        self.assertEqual(self.verify()["status"], "verified")


class JSONAndSourceTests(unittest.TestCase):
    def test_duplicate_json_keys(self):
        with self.assertRaises(ValueError): v.loads(b'{"a":1,"a":2}')

    def test_nonfinite_json(self):
        for value in (b"NaN", b"Infinity", b"-Infinity"):
            with self.assertRaises(ValueError): v.loads(value)

    def test_bounded_read(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data"; p.write_bytes(b"abc")
            with self.assertRaises(ValueError): v.read_bytes(p, 2)

    def test_312_syntax(self):
        ast.parse((HERE / "natural_keyword_evidence_v2.py").read_text(), feature_version=(3,12))

    def test_no_assert_security_checks(self):
        tree = ast.parse((HERE / "natural_keyword_evidence_v2.py").read_text())
        self.assertFalse(any(isinstance(n, ast.Assert) for n in ast.walk(tree)))

    def test_no_model_or_network_imports(self):
        tree = ast.parse((HERE / "natural_keyword_evidence_v2.py").read_text())
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import): names.update(a.name.split('.')[0] for a in n.names)
            if isinstance(n, ast.ImportFrom): names.add((n.module or '').split('.')[0])
        self.assertFalse(names & {"torch", "requests", "http", "urllib", "socket", "subprocess", "transformers"})

    def test_cli_refuses_wrong_runtime(self):
        with patch.object(v.sys, "version_info", (3, 13, 5)), patch.object(v, "_replay") as replay:
            self.assertEqual(v.main(["create", "--upstream", "/not-read", "--upstream-manifest-sha256", pin("x"),
                                     "--output", "/not-created", "--audit-salt", "x"*16]), 2)
            replay.assert_not_called()


if __name__ == "__main__":
    unittest.main()
