"""Replication transport boundaries with invented bytes; no source/model calls."""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_replication_transport as replica
import english_v6_b21_replication_source_producer as producer


class Response(io.BytesIO):
    def __init__(self, data, url, start, end, total):
        super().__init__(data)
        self.status, self.url = 206, url
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}",
                        "Content-Length": str(end - start + 1)}

    def geturl(self):
        return self.url


class Opener:
    def __init__(self, data, calls):
        self.data, self.calls = data, calls

    def open(self, request, timeout):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        self.calls.append((request.full_url, start, end))
        return Response(self.data[start:end + 1], request.full_url, start, end, len(self.data))


@contextmanager
def fixture(valid_hash=True):
    data = bytes(range(256)) * 8
    identity = dict(run_id="123", run_head="2" * 40, prepared_parent="1" * 40,
        run_attempt=1, run_branch=replica.lineage.BRANCH)
    permission = dict(source_execution=identity, primary_quality_lock_sha256="a" * 64)
    spec = replica.plan()
    spec.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest() if valid_hash else "0" * 64)
    upstream = dict(bytes=spec["bytes"], sha256=spec["sha256"])
    calls = []
    with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
        root = Path(temporary); prior = root / "prior"; prior.mkdir()
        corpus = root / "train-clean-100.tar.gz"
        stack.enter_context(patch.object(replica.lineage, "execution_guard", return_value=identity))
        stack.enter_context(patch.object(replica.lineage, "require_authorization", return_value=permission))
        stack.enter_context(patch.object(replica.lineage, "require_uploaded_opening", return_value=permission))
        stack.enter_context(patch.object(replica.lineage, "read_plan", return_value={"replication": {"natural": {"upstream": upstream}}}))
        stack.enter_context(patch.object(replica.transport, "plan", return_value=spec))
        stack.enter_context(patch.object(replica.transport.urllib.request, "build_opener", side_effect=lambda *args: Opener(data, calls)))
        yield root, prior, corpus, data, calls, identity, permission


class ReplicationTransport(unittest.TestCase):
    def test_uses_frozen_fixed_mirror_and_exact_same_corpus(self):
        spec = replica.plan()
        self.assertEqual(spec["urls"], [replica.transport.OFFICIAL[3]])
        self.assertEqual((spec["parallel_ranges"], spec["range_retries"], spec["fallback_urls"]), (8, 0, []))
        self.assertEqual(replica.digest(Path(replica.transport.__file__).read_bytes()), replica.SHARED_TRANSPORT_SHA)
        with patch.object(replica.lineage, "read_opening", return_value={"primary_quality_lock": None}):
            self.assertEqual(replica.check()["status"], "INERT_REQUIRES_ORIGINAL_PRIMARY_PASS")
        self.assertNotIn("torch", sys.modules); self.assertNotIn("numpy", sys.modules)

    def test_sealed_primary_stops_before_transport_or_directories(self):
        value = json.loads(replica.lineage.OPENING.read_bytes())
        value.update(status="SEALED_UNTIL_ORIGINAL_PRIMARY_PASS", quality_readback_sha256=None, primary_quality_lock=None)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); opening = root / "opening.json"; opening.write_text(json.dumps(value))
            with patch.object(replica.lineage, "OPENING", opening), \
                 patch.object(replica.transport.urllib.request, "build_opener", side_effect=AssertionError("no transport")) as network:
                for action in (replica.download, replica.verify):
                    with self.assertRaisesRegex(ValueError, "replication remains sealed"):
                        action(root / "train-clean-100.tar.gz", root / "prior", "1" * 40)
                network.assert_not_called()
            self.assertEqual([p.name for p in root.iterdir()], ["opening.json"])

    def test_authorization_for_another_event_stops_before_requests(self):
        with fixture() as (_, prior, corpus, _, calls, identity, permission):
            altered = dict(permission, source_execution=dict(identity, run_id="124"))
            with patch.object(replica.lineage, "require_authorization", return_value=altered):
                with self.assertRaisesRegex(ValueError, "exact first replication event"):
                    replica.download(corpus, prior, "1" * 40)
            self.assertEqual(calls, []); self.assertFalse((prior / "TRANSPORT").exists())

    def test_eight_real_workers_assemble_exact_bytes_and_cannot_resume(self):
        with fixture() as (_, prior, corpus, data, calls, identity, _):
            result = replica.download(corpus, prior, "1" * 40)
            self.assertEqual(result["status"], "COMPLETE_VERIFIED")
            self.assertEqual(result["panel"], "replication")
            self.assertEqual(result["source_execution"], identity)
            self.assertEqual(corpus.read_bytes(), data)
            self.assertEqual(len(calls), 8)
            self.assertEqual({url for url, _, _ in calls}, {replica.transport.OFFICIAL[3]})
            ordered = sorted((start, end) for _, start, end in calls)
            self.assertEqual(ordered[0][0], 0); self.assertEqual(ordered[-1][1] + 1, len(data))
            self.assertTrue(all(a[1] + 1 == b[0] for a, b in zip(ordered, ordered[1:])))
            self.assertEqual(replica.verify(corpus, prior, "1" * 40), result)
            with self.assertRaisesRegex(ValueError, "no resume or retry"):
                replica.download(corpus, prior, "1" * 40)
            self.assertEqual(len(calls), 8)

    def test_wrong_whole_hash_preserves_stop_without_ready_corpus(self):
        with fixture(valid_hash=False) as (_, prior, corpus, _, calls, _, _):
            with self.assertRaisesRegex(ValueError, "every original compressed byte"):
                replica.download(corpus, prior, "1" * 40)
            self.assertFalse(corpus.exists()); self.assertTrue(corpus.with_name(corpus.name + ".partial").exists())
            self.assertFalse((prior / "CORPUS_TRANSPORT.json").exists())
            self.assertEqual(json.loads((prior / "TRANSPORT/RESULT.json").read_bytes())["status"], "STOP")
            self.assertEqual(len(calls), 8)

    def test_verified_bytes_cannot_reuse_another_primary_lock(self):
        with fixture() as (_, prior, corpus, _, _, _, permission):
            replica.download(corpus, prior, "1" * 40)
            with patch.object(replica.lineage, "require_authorization", return_value=dict(permission, primary_quality_lock_sha256="b" * 64)):
                with self.assertRaisesRegex(ValueError, "authorized replication event"):
                    replica.verify(corpus, prior, "1" * 40)

    def test_template_keeps_pass_before_transport_and_uploaded_receipts_before_sources(self):
        root = Path(__file__).resolve().parents[1]
        template = (root / "automation/english_v6_b21_replication_source_workflow_template.yml").read_text()
        actual = (root / replica.lineage.WORKFLOW).read_text()
        parent = re.search(r"PREPARED_PARENT: '([^']+)'", actual).group(1)
        self.assertEqual(actual, template.replace("__B21_REPLICATION_SOURCE_MAIN_SHA__", parent))
        self.assertNotIn("workflow_dispatch", template)
        self.assertNotIn("curl --fail", template)
        authorize = template.index("english_v6_b21_replication_source.py authorize ")
        download = template.index("english_v6_b21_replication_transport.py download ")
        upload = template.index("name: english-v6-b21-replication-corpus-transport-")
        verify = template.index("english_v6_b21_replication_transport.py verify ")
        generate = template.index("english_v6_b21_replication_source.py generate ")
        self.assertLess(authorize, download); self.assertLess(download, upload)
        self.assertLess(upload, verify); self.assertLess(verify, generate)
        python = producer.common.prior.static_closure(root / "automation", producer.ROOTS)
        for name in ("english_v6_b21_replication_transport", "english_v6_b21_fixed_mirror_transport",
                     "english_v6_b21_fixed_mirror_quality_readback", "english_v6_b21_fixed_mirror_primary_eval"):
            self.assertIn("automation/" + name + ".py", python)


if __name__ == "__main__":
    unittest.main()
