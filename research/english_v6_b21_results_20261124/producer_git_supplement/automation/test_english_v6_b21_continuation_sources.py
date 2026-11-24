"""Audited continuation/transport boundaries; invented buffers, never audio/models."""
from __future__ import annotations
import ast
import copy
from contextlib import contextmanager, ExitStack
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_continuation_source as source
import english_v6_b21_continuation_source_ancestry as lineage
import english_v6_b21_continuation_source_producer as producer
import english_v6_b21_continuation_transport as transport


class FakeResponse(io.BytesIO):
    def __init__(self, data, url, start, end, total):
        super().__init__(data)
        self.status, self.url = 206, url
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}", "Content-Length": str(end - start + 1)}

    def geturl(self): return self.url


class FakeOpener:
    def __init__(self, data, calls): self.data, self.calls = data, calls

    def open(self, request, timeout):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
        self.calls.append((request.full_url, start, end))
        return FakeResponse(self.data[start:end + 1], request.full_url, start, end, len(self.data))


class Continuation(unittest.TestCase):
    def test_metadata_and_scientific_operators_remain_exactly_frozen(self):
        source.check(); spec = transport.plan()
        self.assertEqual(source.source_plan(), source.primary.source_plan())
        self.assertEqual(source.source_plan("replication"), source.primary.source_plan("replication"))
        self.assertEqual(spec["urls"], list(transport.OFFICIAL))
        self.assertEqual((spec["parallel_ranges"], spec["range_retries"]), (8, 0))
        def functions(module):
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(Path(module.__file__).read_text()).body if isinstance(node, ast.FunctionDef)}
        current, frozen = functions(source.observers), functions(source.observers.primary_observer)
        for name in ("validate_rows", "observe", "load_observations"):
            self.assertEqual(current[name], frozen[name])
        self.assertEqual(functions(source)["screen_document"], functions(source.primary)["screen_document"])
        contract = source.code_contract()
        self.assertEqual(contract["source_indices"], list(range(846, 870)))
        self.assertEqual(contract["frozen_primary_observer_sha256"], source.primary.code_contract()["observer_sha256"])
        self.assertNotIn("torch", sys.modules); self.assertNotIn("numpy", sys.modules)

    def test_missing_original_failure_proof_stops_every_runtime_before_network_or_directory(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(lineage.origin, "verify", side_effect=ValueError("original failure proof missing")), \
             patch.object(lineage, "gh", side_effect=AssertionError("must not call network")), \
             patch.object(source.B13, "render_panel", side_effect=AssertionError("must not synthesize")):
            root = Path(tmp)
            for action in (
                lambda: lineage.authorize(root / "authorization", "1" * 40),
                lambda: transport.download(root / "train-clean-100.tar.gz", root / "authorization", "1" * 40),
                lambda: source.generate(root / "source", root / "authorization", root / "corpus", "1" * 40),
                lambda: source.observe("xlsr", root / "source", root / "native", "1" * 40),
            ):
                with self.assertRaisesRegex(ValueError, "original failure proof missing"): action()
            self.assertEqual(list(root.iterdir()), [])

    def test_first_new_public_workflow_only_identity_and_no_original_branch_replay(self):
        parent, head = "1" * 40, "2" * 40
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=lineage.REPO, GITHUB_REF=lineage.BRANCH,
            GITHUB_EVENT_NAME="push", GITHUB_RUN_ATTEMPT="1", GITHUB_RUN_ID="123", GITHUB_SHA=head)
        event = dict(before=parent, after=head, ref=lineage.BRANCH, forced=False, deleted=False,
            repository=dict(full_name=lineage.REPO, private=False))
        identity = lineage.execution_guard(parent, env, event, parent)
        self.assertEqual(identity["run_branch"], lineage.BRANCH)
        self.assertEqual(producer.validate_event(parent, head, [parent], [lineage.WORKFLOW], env, event)["role"], "source_continuation")
        for bad_env in (dict(env, GITHUB_RUN_ATTEMPT="2"), dict(env, GITHUB_REF=lineage.frozen.BRANCH)):
            with self.assertRaises(ValueError): lineage.execution_guard(parent, bad_env, event, parent)
        with self.assertRaises(ValueError): lineage.execution_guard(parent, env, dict(event, forced=True), parent)
        with self.assertRaises(ValueError): producer.validate_event(parent, head, [parent], [lineage.WORKFLOW, "automation/changed.py"], env, event)

    def test_prior_authorization_is_exact_original_opened_marker_not_same_run_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); shutil.copytree(lineage.origin.FAILURE_DIR, root / "TRANSPORT_ORIGIN")
            prior = root / "PRIOR"; prior.mkdir()
            with lineage.zipfile.ZipFile(root / "TRANSPORT_ORIGIN/opened-original.zip") as archive:
                original = json.loads(archive.read("AUTHORIZATION.json"))
            original["source_execution"]["prepared_parent"] = "1" * 40
            (prior / "AUTHORIZATION.json").write_text(json.dumps(original))
            with patch.object(lineage.frozen, "require_authorization", side_effect=AssertionError("wrong original marker must stop first")):
                with self.assertRaisesRegex(ValueError, "exact original opened-marker member"):
                    lineage.require_authorization(root)

    def test_corpus_or_consumption_gate_failure_cannot_enter_renderer(self):
        identity = dict(run_id="123", run_head="2" * 40, prepared_parent="1" * 40, run_attempt=1, run_branch=lineage.BRANCH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prior = root / "prior"; prior.mkdir(); (prior / "HISTORY.json").write_text("{}")
            with patch.object(lineage, "execution_guard", return_value=identity), \
                 patch.object(lineage, "require_authorization", return_value={"source_execution": identity}), \
                 patch.object(lineage, "validate_history", return_value=(set(), set(), set())), \
                 patch.object(lineage, "require_generation_permit", side_effect=ValueError("whole corpus or durable marker missing")), \
                 patch.object(source.B13, "render_panel") as render, patch.object(source.natural, "extract") as decode:
                with self.assertRaisesRegex(ValueError, "whole corpus or durable marker missing"):
                    source.generate(root / "source", prior, root / "corpus", "1" * 40)
                render.assert_not_called(); decode.assert_not_called(); self.assertFalse((root / "source").exists())

    def test_eight_disjoint_ranges_exact_headers_and_preserved_redirect_range(self):
        parts = transport.ranges(lineage.CORPUS_BYTES, 8)
        self.assertEqual(parts[0]["start"], 0); self.assertEqual(parts[-1]["end"] + 1, lineage.CORPUS_BYTES)
        self.assertTrue(all(a["end"] + 1 == b["start"] for a, b in zip(parts, parts[1:])))
        self.assertEqual(sum(p["end"] - p["start"] + 1 for p in parts), lineage.CORPUS_BYTES)
        p = parts[0]; headers = {"Content-Range": f'bytes {p["start"]}-{p["end"]}/{lineage.CORPUS_BYTES}', "Content-Length": str(p["end"] + 1)}
        transport.validate_headers(206, headers, p, lineage.CORPUS_BYTES)
        for status, changed in ((200, headers), (206, dict(headers, **{"Content-Range": "bytes 0-4/5"})),
                                (206, dict(headers, **{"Content-Length": "1"})), (206, dict(headers, **{"Content-Encoding": "gzip"}))):
            with self.assertRaises(ValueError): transport.validate_headers(status, changed, p, lineage.CORPUS_BYTES)
        original = transport.OFFICIAL[3]
        request = transport.urllib.request.Request(original, headers={"Range": "bytes=5-9", "Accept-Encoding": "identity"})
        redirected = transport.OfficialRedirect(original).redirect_request(request, None, 302, "found", {}, "https://huggingface.co/example")
        self.assertEqual(redirected.get_header("Range"), "bytes=5-9")
        for url in ("http://huggingface.co/example", "https://example.org/corpus"):
            with self.assertRaises(ValueError): transport.OfficialRedirect(original).redirect_request(request, None, 302, "found", {}, url)
        self.assertFalse(transport.final_url_allowed(transport.OFFICIAL[0], "https://huggingface.co/example"))

    def test_direct_partial_file_atomic_success_and_bad_whole_hash_stays_closed(self):
        payload = bytes(range(256)) * 8
        identity = dict(run_id="123", run_head="2" * 40, prepared_parent="1" * 40, run_attempt=1, run_branch=lineage.BRANCH)
        for good in (True, False):
            with self.subTest(whole_hash_valid=good), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); prior = root / "prior"; prior.mkdir(); corpus = root / "train-clean-100.tar.gz"
                spec = transport.plan(); spec["bytes"] = len(payload)
                spec["sha256"] = hashlib.sha256(payload).hexdigest() if good else "0" * 64
                calls = []
                with patch.object(lineage, "execution_guard", return_value=identity), \
                     patch.object(lineage, "require_authorization", return_value={"source_execution": identity, "transport_origin": {"fixture": True}}), \
                     patch.object(lineage, "one_execution"), patch.object(transport, "plan", return_value=spec), \
                     patch.object(transport.urllib.request, "build_opener", side_effect=lambda *args: FakeOpener(payload, calls)):
                    if good:
                        result = transport.download(corpus, prior, "1" * 40)
                        self.assertEqual(result["status"], "COMPLETE_VERIFIED")
                        self.assertEqual(corpus.read_bytes(), payload)
                        self.assertFalse(corpus.with_name(corpus.name + ".partial").exists())
                    else:
                        with self.assertRaisesRegex(ValueError, "every original compressed byte"):
                            transport.download(corpus, prior, "1" * 40)
                        self.assertFalse(corpus.exists())
                        self.assertTrue(corpus.with_name(corpus.name + ".partial").exists())
                        self.assertFalse((prior / "CORPUS_TRANSPORT.json").exists())
                        self.assertEqual(json.loads((prior / "TRANSPORT/RESULT.json").read_bytes())["status"], "STOP")
                self.assertEqual(len(calls), 8)
                self.assertEqual([sum(url == official for url, _, _ in calls) for official in transport.OFFICIAL], [2, 2, 2, 2])

    @contextmanager
    def observer_fixture(self):
        rows = copy.deepcopy(source.source_plan()["rows"])
        for i in range(24):
            rows.append(dict(id="b21-primary-natural-fixture-" + str(i), cohort="natural", role="development",
                training_allowed=False, observed_class=None, requested_class=0, whole_original=True))
        for i, row in enumerate(rows):
            value = hashlib.sha256(("fake-not-audio-" + str(i)).encode()).hexdigest()
            row.update(pcm_sha256=value, wav_sha256=value, samples=16000)
        identity = dict(run_id="123", run_head="2" * 40, prepared_parent="1" * 40, run_attempt=1, run_branch=lineage.BRANCH)
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); src = root / "source"; src.mkdir(); (src / "MANIFEST.json").write_text("{}")
            stack.enter_context(patch.object(lineage, "execution_guard", return_value=identity))
            stack.enter_context(patch.object(source, "_read", return_value=(rows, rows, {}, {"source_execution": identity})))
            yield root, src, rows, source.registration(src)

    def test_actual_continuation_owner_guard_observe120_and_readback_front(self):
        import english_hifi_micro as micro
        observer = source.observers
        with self.observer_fixture() as (root, src, rows, registration):
            by_id = {row["id"]: row for row in rows}; calls = []
            def backend(wav, directory, start):
                calls.append(wav.stem)
                for name in ("FORWARD_STARTED.json", "FORWARD_RETURNED.json", "RAW_PHONE.json", "PAYLOAD_VALID.json"):
                    source.qa.record(directory / name, {"invented_fixture": True})
                return {"invented_fixture": True}
            with patch.object(observer, "_load_phone", return_value=(backend, {"fixture": True})), \
                 patch.object(micro, "inspect_wav", side_effect=lambda path: by_id[path.stem]):
                result = source.observe("xlsr", src, root / "native", "1" * 40)
            self.assertEqual((len(calls), result["valid_native_payloads"], result["status"]), (120, 120, "COMPLETE"))
            ledger = observer.row_ledger(root / "native", rows)
            self.assertTrue(observer.complete_ledger(ledger, 120)); self.assertFalse(observer.complete_ledger(ledger[:-1], 120))
            with self.assertRaises(FileNotFoundError) as missing:
                observer.load_observations(src, root / "missing-providers", rows, source.BATCH, registration=registration)
            self.assertTrue(str(missing.exception.filename).endswith("FILES.json"))
            with self.assertRaises(ValueError): observer.load_observations(src, root, rows, source.primary.BATCH, registration=registration)

    def test_closed_workflow_order_required_tokens_and_all_original_proof_resources(self):
        root = Path(__file__).resolve().parents[1]
        template = root / "automation/english_v6_b21_continuation_source_workflow_template.yml"
        text = template.read_text()
        actual = (root / lineage.WORKFLOW).read_text()
        parent = re.search(r"PREPARED_PARENT: '([^']+)'", actual).group(1)
        self.assertEqual(actual, text.replace("__B21_SOURCE_CONTINUATION_MAIN_SHA__", parent))
        self.assertNotIn("workflow_dispatch", text)
        self.assertEqual(text.count("include-hidden-files: true"), text.count("uses: actions/upload-artifact@"))
        command = lambda name: 'english_v6_b21_continuation_source.py ' + name + ' '
        self.assertLess(text.index(command('authorize')), text.index(command('transport')))
        self.assertLess(text.index(command('transport')), text.index(command('consume')))
        self.assertLess(text.index('name: ' + lineage.MARKER), text.index(command('generate')))
        self.assertLess(text.index(command('consume')), text.index('piper-tts=='))
        sections = text.split('    - name: ')
        for name in ('authorize', 'transport', 'consume', 'generate'):
            section = next(s for s in sections if command(name) in s)
            self.assertIn('GH_TOKEN: ${{ github.token }}', section)
        python = producer.common.prior.static_closure(root / "automation", producer.ROOTS)
        self.assertIn("automation/english_v6_b21_continuation_origin.py", python)
        self.assertIn("automation/english_v6_b21_continuation_transport.py", python)
        self.assertEqual(len([p for p in lineage.origin.FAILURE_DIR.rglob("*") if p.is_file()]), 10)
        self.assertEqual(len([p for p in transport.PROBE.rglob("*") if p.is_file()]), 4)


if __name__ == "__main__": unittest.main()
