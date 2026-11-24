"""Exact reader routing and closed downstream gates; no acoustic execution."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import english_v6_b21_fixed_mirror_quality_readback as reader
import english_v6_b21_replication_eval as replica
import english_v6_b21_replication_source_ancestry as source
import english_v6_b21_engineering as engineering


class Routing(unittest.TestCase):
    def test_every_original_quality_function_is_unchanged(self):
        def functions(path):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(path.read_text()).body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        self.assertEqual(functions(HERE / "english_v6_b21_quality_readback.py"), functions(Path(reader.__file__)))
        self.assertEqual(reader.sha((HERE / "english_v6_b21_fixed_mirror_primary_eval.py").read_bytes()), reader.PRIMARY_CODE_SHA)
        self.assertEqual(reader.EVALUATOR_STEMS["primary"], "english_v6_b21_fixed_mirror_primary")
        self.assertEqual(reader.EVALUATOR_STEMS["replication"], "english_v6_b21_replication")

    def test_old_continuation_primary_lock_cannot_open_new_reader(self):
        def pin(number):
            return dict(id=number, run_id=123, run_head="1" * 40, bytes=100, sha256="2" * 64)
        lock = dict(schema="english-v6-b21-original-quality-lock-v1", panel="primary",
            result=pin(1), producer=pin(2), consumed=pin(3), evaluator_sha256=reader.PRIMARY_CODE_SHA,
            inputs_sha256="3" * 64, producer_manifest_sha256="4" * 64, original_result_sha256="5" * 64)
        reader.validate_lock(lock)
        lock["evaluator_sha256"] = "79a2a4e9cf28f7da40b985cadfb2e1c06e7fa1c50e52b4e91242f2abc48f7245"
        with self.assertRaisesRegex(ValueError, "unchanged preregistered primary evaluator"):
            reader.validate_lock(lock)

    def test_source_scorer_and_engineering_resolve_identical_reader(self):
        digest = reader.sha(Path(reader.__file__).read_bytes())
        self.assertIs(replica.quality_reader({"quality_readback_sha256": digest}), reader)
        with patch.object(source, "read_opening", return_value={"quality_readback_sha256": digest}):
            self.assertIs(source.quality_reader(), reader)
        self.assertEqual(engineering.QUALITY_READER, reader.__name__)
        self.assertIs(engineering.module(engineering.QUALITY_READER), reader)
        self.assertEqual(json.loads(engineering.SPEC.read_bytes())["quality_reader"], reader.__name__)

    def test_unset_future_locks_remain_closed_without_blocking_later_valid_binding(self):
        opening = json.loads(source.OPENING.read_bytes())
        opening.update(status="SEALED_UNTIL_ORIGINAL_PRIMARY_PASS", primary_quality_lock=None, quality_readback_sha256=None)
        spec = json.loads(replica.SPEC.read_bytes())
        spec.update(scoring_allowed=False, replication_opened=False, primary_quality_lock=None, quality_readback_sha256=None)
        engineering_spec = json.loads(engineering.SPEC.read_bytes())
        engineering_spec.update(enabled=False, quality_reader_sha256=None,
            quality_locks={"primary": None, "replication": None}, candidate_files=None)
        engineering_spec["inputs"]["pair"] = None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, value in (("opening", opening), ("replica", spec), ("engineering", engineering_spec)):
                (root / (name + ".json")).write_text(json.dumps(value))
            with patch.object(source, "OPENING", root / "opening.json"), \
                 patch.object(replica, "SPEC", root / "replica.json"), \
                 patch.object(engineering, "SPEC", root / "engineering.json"):
                self.assertEqual(source.read_opening(require_bound=False)["status"], "SEALED_UNTIL_ORIGINAL_PRIMARY_PASS")
                with self.assertRaisesRegex(ValueError, "replication remains sealed"):
                    source.read_opening()
                with self.assertRaises(ValueError):
                    replica.protocol()
                self.assertEqual(engineering.source_check()["status"], "INERT_AWAIT_ORIGINAL_DUAL_QUALITY_PASS")
        self.assertEqual(engineering.MAX_CALLS, 140)
        self.assertEqual(replica.MAX_FORWARDS, 1440)


if __name__ == "__main__":
    unittest.main()
