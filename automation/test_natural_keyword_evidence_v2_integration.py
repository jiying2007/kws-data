#!/usr/bin/env python3
"""Real verifier composition on fictional PCM/native evidence, never ML calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "automation")]
from automation import natural_keyword_evidence_v2 as v
from automation import active_natural_asr_weak_view as natural
from test_active_natural_asr_weak_view import NaturalFixture, write_json


def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class NativeVerifierIntegration(unittest.TestCase):
    """No mock: production create/verify must replay real upstream test format."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.upstream = self.root / "upstream"
        self.output = self.root / "keyword"
        self.fixture = NaturalFixture(self.root / "inputs", texts=[
            ("今天你好小窝呀", "你好小窝天气不错", "我说你好小窝"),
            ("今天晴天", "明天下雨", "请打开灯")])

    def prepare(self):
        request = self.root / "request.json"
        write_json(request, {"schema": natural.REQUEST_SCHEMA,
                             "batches": [self.fixture.entry()]})
        natural.create_view(request, v.digest(request.read_bytes()), self.upstream)
        self.upstream_sha = v.digest((self.upstream / "MANIFEST.json").read_bytes())
        return natural.verify_view(self.upstream, expected_manifest_sha256=self.upstream_sha)

    def create(self):
        return v.create(self.upstream, self.upstream_sha, self.output,
                        salt="v3-real-format-integration-fixture", limit=2)

    def verify(self):
        return v.verify(self.upstream, self.upstream_sha, self.output,
                        v.digest((self.output / "MANIFEST.json").read_bytes()))

    def test_actual_native_verifier_then_keyword_projection(self):
        original = self.prepare()
        before = snapshot(self.upstream)
        self.create()
        report = json.loads((self.output / "report.json").read_bytes())
        self.assertEqual([x["machine_candidate_state"] for x in
                          report["rows"][0]["keyword_candidates"]], ["present", "unknown"])
        self.assertEqual([x["machine_candidate_state"] for x in
                          report["rows"][1]["keyword_candidates"]], ["unknown", "unknown"])
        for old, new in zip(original["rows"], report["rows"]):
            self.assertEqual(old["decision"], new["original_fulltext_decision"])
        self.assertEqual(self.verify()["status"], "verified")
        self.assertEqual(before, snapshot(self.upstream))
        self.assertFalse(report["boundary"]["training_admitted"])
        self.assertEqual(report["boundary"]["accepted_negative_seconds"], 0)

    def test_inherited_fulltext_labels_are_unchanged(self):
        self.fixture.texts = [("你好小窝",) * 3, ("请开灯",) * 3]
        for clip, texts in zip(self.fixture.inputs["clips"], self.fixture.texts):
            for model, text in zip(v.MODELS, texts):
                self.fixture.native(clip, model, text)
        self.fixture.seal_fixture()
        original = self.prepare()
        self.create()
        report = json.loads((self.output / "report.json").read_bytes())
        for old, new in zip(original["rows"], report["rows"]):
            self.assertEqual(new["original_fulltext_decision"], old["decision"])
            self.assertTrue(all(x["source"] == "inherited" for x in new["keyword_candidates"]))
            self.assertTrue(all(x["training_mask"] is False for x in new["keyword_candidates"]))
        self.verify()

    def test_quality_flag_prevents_keyword_candidate(self):
        self.fixture.native(self.fixture.inputs["clips"][0], "whisper",
                            "我说你好小窝", quality=["fixture-warning"], terminated=False)
        self.fixture.seal_fixture()
        self.prepare()
        self.create()
        report = json.loads((self.output / "report.json").read_bytes())
        self.assertTrue(all(x["machine_candidate_state"] == "unknown"
                            for x in report["rows"][0]["keyword_candidates"]))
        self.verify()

    def test_missing_actual_seal_fails_without_v2_output(self):
        self.prepare()
        seals = list(self.upstream.glob("batches/**/final-summary.json"))
        self.assertTrue(seals)
        seals[0].unlink()
        with self.assertRaises((ValueError, OSError, KeyError)):
            self.create()
        self.assertFalse(self.output.exists())

    def test_tampered_native_bytes_fail_without_v2_output(self):
        self.prepare()
        native = list(self.upstream.glob("batches/**/native/qwen06/*.json"))
        self.assertTrue(native)
        raw = json.loads(native[0].read_bytes())
        raw["native"]["decoded_skip_special_tokens"] = "language Chinese<asr_text>伪造文本"
        native[0].write_text(json.dumps(raw, ensure_ascii=False))
        with self.assertRaises((ValueError, OSError, KeyError)):
            self.create()
        self.assertFalse(self.output.exists())

    def test_v2_forgery_and_rehash_cannot_replace_native_evidence(self):
        self.prepare()
        self.create()
        report_path = self.output / "report.json"
        report = json.loads(report_path.read_bytes())
        report["rows"][0]["keyword_candidates"][1]["machine_candidate_state"] = "absent"
        report_path.write_bytes(v.canonical(report) + b"\n")
        path = self.output / "MANIFEST.json"
        manifest = json.loads(path.read_bytes())
        manifest["files"]["report.json"] = v.record(report_path.read_bytes())
        path.write_bytes(v.canonical(manifest) + b"\n")
        with self.assertRaisesRegex(ValueError, "NOT_DERIVED_FROM_NATIVE_EVIDENCE"):
            self.verify()

    def test_retained_evidence_replays_after_original_paths_are_removed(self):
        self.prepare()
        self.create()
        shutil.rmtree(self.fixture.root)
        moved = self.root / "moved-upstream"
        self.upstream.rename(moved)
        self.upstream = moved
        before = snapshot(moved)
        self.assertEqual(self.verify()["status"], "verified")
        self.assertEqual(before, snapshot(moved))


if __name__ == "__main__":
    unittest.main()
