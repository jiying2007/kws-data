#!/usr/bin/env python3
"""Artificial request/download/receipt tests; no speech, model or network calls."""

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import urllib.error
import zipfile

SPEC = importlib.util.spec_from_file_location("librispeech_consumer_under_test",
                                           Path(__file__).with_name("research_v4_native_c_librispeech.py"))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class Response(io.BytesIO):
    def __init__(self, value, url, status=200, length=None):
        super().__init__(value)
        self.url, self.status = url, status
        self.headers = {} if length is None else {"Content-Length": length}

    def geturl(self):
        return self.url


class ConsumerContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / (C.REQUEST_ID + ".json")
        self.request = copy.deepcopy(C.FIXED)
        self.request.update(pipeline_commit=C.PIPELINE_COMMIT, reason="Invented source-only fixture; no actual speech.")
        self.write(self.path, self.request)
        self.digest = C.core.sha256(self.path.read_bytes())
        self.event = self.root / "event.json"
        self.write(self.event, {"repository": {"private": False}})

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")

    def environment(self, **changes):
        values = {"GITHUB_REPOSITORY": C.core.REPOSITORY, "GITHUB_EVENT_NAME": "push",
                  "GITHUB_REF": "refs/heads/main", "GITHUB_EVENT_PATH": str(self.event),
                  "GITHUB_RUN_ID": "999", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_ACTIONS": "true",
                  "GITHUB_SHA": "b" * 40}
        values.update(changes)
        return mock.patch.dict(os.environ, values, clear=True)

    def execution_fixture(self):
        state, pipeline = self.root / "state", self.root / "pipeline"
        (state / "result").mkdir(parents=True)
        (state / "claim").mkdir()
        (state / "result/REQUEST.json").write_bytes(self.path.read_bytes())
        script = pipeline / C.ENTRYPOINT
        script.parent.mkdir(parents=True)
        script.write_text("# invented entrypoint, never executed\n")
        self.write(state / "claim/CLAIM.json", C.claim_record(self.request, self.digest,
                   {"run_id": "999", "run_attempt": "1"}))
        return state, pipeline, script

    def test_fixed_request_accepts_only_exact_known_fields_and_scalar_types(self):
        self.assertEqual(C.load_request(self.path)[2], self.digest)
        for key, value in (("cpu_threads", 2), ("optimizer_updates", False), ("optimizer_updates", 1),
                           ("human_gold", True), ("new_independent_audio_seconds", 1),
                           ("max_execution_seconds", 901), ("max_verification_seconds", 181),
                           ("pipeline_commit", "main"), ("pipeline_commit", "a" * 40), ("entrypoint", "arbitrary.py"),
                           ("request_id", "another-request"), ("reason", "")):
            with self.subTest(key=key, value=value):
                request = copy.deepcopy(self.request)
                request[key] = value
                self.write(self.path, request)
                with self.assertRaises(ValueError):
                    C.load_request(self.path)
        for key, value in (("full_passes", 3), ("native_child_seconds", 181), ("pool_time_major", 1),
                           ("source_pcm_samples", 58646164.0), ("models", ["dsc_tcn", "gru"])):
            request = copy.deepcopy(self.request)
            request["profile"][key] = value
            self.write(self.path, request)
            with self.subTest(profile=key), self.assertRaises(ValueError):
                C.load_request(self.path)

    def test_request_duplicate_extra_filename_and_budget_changes_fail(self):
        self.path.write_text('{"schema":"x","schema":"y"}')
        with self.assertRaises(ValueError):
            C.load_request(self.path)
        request = dict(self.request, shell="echo arbitrary")
        self.write(self.path, request)
        with self.assertRaises(ValueError):
            C.load_request(self.path)
        other = self.root / "another.json"
        self.write(other, self.request)
        with self.assertRaises(ValueError):
            C.load_request(other)
        request = copy.deepcopy(self.request)
        request["source_artifacts"]["corpus"]["url"] = "https://unapproved.invalid/audio"
        self.write(self.path, request)
        with self.assertRaises(ValueError):
            C.load_request(self.path)

    def test_download_is_single_fixed_unauthenticated_create_only_transfer(self):
        raw = b"invented corpus archive, not audio"
        values = {"bytes": len(raw), "sha256": C.core.sha256(raw)}
        target = self.root / "corpus.tar.gz"
        opener = mock.Mock()
        opener.open.return_value = Response(raw, C.CORPUS["url"], length=str(len(raw)))
        with mock.patch.dict(C.CORPUS, values), mock.patch.object(C.urllib.request, "build_opener", return_value=opener):
            C.download_corpus(target)
            with self.assertRaises(ValueError):
                C.download_corpus(target)
        self.assertEqual(target.read_bytes(), raw)
        opener.open.assert_called_once()
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, C.CORPUS["url"])
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(opener.open.call_args.kwargs, {"timeout": 30})

    def test_download_rejects_truncation_oversize_hash_redirect_status_and_length(self):
        raw = b"invented corpus"
        for case in ("short", "long", "digest", "url", "status", "length", "timeout"):
            target = self.root / (case + ".tar.gz")
            observed = raw[:-1] if case == "short" else raw + b"x" if case == "long" else raw
            response = Response(observed, "https://changed.invalid" if case == "url" else C.CORPUS["url"],
                                status=206 if case == "status" else 200,
                                length="1" if case == "length" else None)
            opener = mock.Mock()
            opener.open.return_value = response
            if case == "timeout":
                opener.open.side_effect = TimeoutError("invented timeout")
            with self.subTest(case=case), mock.patch.dict(C.CORPUS, {
                "bytes": len(raw), "sha256": "f" * 64 if case == "digest" else C.core.sha256(raw)}), \
                mock.patch.object(C.urllib.request, "build_opener", return_value=opener):
                with self.assertRaises((ValueError, TimeoutError)):
                    C.download_corpus(target)
            self.assertFalse(target.exists())
            self.assertTrue(target.with_suffix(".part").exists())
            opener.open.assert_called_once()

    def test_download_stops_at_absolute_deadline_without_retry(self):
        target = self.root / "time.tar.gz"
        opener = mock.Mock()
        opener.open.return_value = Response(b"x", C.CORPUS["url"])
        with mock.patch.dict(C.CORPUS, {"bytes": 1, "sha256": C.core.sha256(b"x")}), \
             mock.patch.object(C.urllib.request, "build_opener", return_value=opener), \
             mock.patch.object(C.time, "monotonic", side_effect=[0, 301]):
            with self.assertRaisesRegex(ValueError, "wall budget"):
                C.download_corpus(target)
        self.assertFalse(target.exists())
        opener.open.assert_called_once()

    def test_consumed_or_ambiguous_requests_and_unproven_rerun_stop_before_download(self):
        cases = (([[], [{"id": 1}]], "1"), ([[{}, {}], []], "1"), ([[], []], "2"))
        for index, (artifacts, attempt) in enumerate(cases):
            with self.subTest(index=index), self.environment(GITHUB_RUN_ATTEMPT=attempt), \
                 mock.patch.object(C.core, "named_artifacts", side_effect=artifacts), \
                 mock.patch.object(C, "download_corpus", side_effect=AssertionError("must not download")), \
                 mock.patch.object(C.native, "run_bounded", side_effect=AssertionError("must not execute")):
                with self.assertRaises(ValueError):
                    C.prepare(self.path, self.root / ("state" + str(index)))

    def test_prior_success_is_saved_only_reuse_and_failed_terminal_never_reexecutes(self):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as archive:
            archive.writestr("TERMINAL.json", '{"invented": true}')
        for failed in (False, True):
            with self.subTest(failed=failed), self.environment(GITHUB_RUN_ATTEMPT="2"), \
                 mock.patch.object(C.core, "named_artifacts", side_effect=[[{"id": 77}], []]), \
                 mock.patch.object(C.core, "artifact_bytes", return_value=out.getvalue()), \
                 mock.patch.object(C, "validate_terminal", side_effect=ValueError("failed consumed terminal") if failed else None) as saved, \
                 mock.patch.object(C, "download_corpus", side_effect=AssertionError("reuse must not download corpus")), \
                 mock.patch.object(C.core, "github_get", side_effect=AssertionError("reuse must not load ancestors")), \
                 mock.patch.object(C.native, "run_bounded", side_effect=AssertionError("reuse must not run models")), \
                 mock.patch.object(C.core, "output") as output:
                state = self.root / ("reused-failed" if failed else "reused-success")
                if failed:
                    with self.assertRaisesRegex(ValueError, "failed consumed"):
                        C.prepare(self.path, state)
                    output.assert_not_called()
                else:
                    C.prepare(self.path, state)
                    self.assertIs(output.call_args.kwargs["execute"], False)
                    self.assertEqual(output.call_args.kwargs["prior_artifact_id"], 77)
                saved.assert_called_once()

    def test_private_wrong_event_and_branch_stop_before_lookup(self):
        for index, change in enumerate(({"GITHUB_EVENT_NAME": "pull_request"}, {"GITHUB_REF": "refs/heads/feature"},
                                         {"GITHUB_REPOSITORY": "another/repo"})):
            with self.environment(**change), mock.patch.object(C.core, "named_artifacts", side_effect=AssertionError("lookup")):
                with self.assertRaises(ValueError):
                    C.prepare(self.path, self.root / ("state" + str(index)))
        self.write(self.event, {"repository": {"private": True}})
        with self.environment(), self.assertRaises(ValueError):
            C.prepare(self.path, self.root / "private")

    def test_prepare_checks_two_fixed_successful_ancestors_and_corpus_before_claim(self):
        def archive_bytes(name, raw):
            out = io.BytesIO()
            with zipfile.ZipFile(out, "w") as archive:
                archive.writestr(name, raw)
            return out.getvalue()
        state_raw = archive_bytes("invented-state.json", b"{}")
        negative_member = b'{"invented": true}'
        negative_raw = archive_bytes(C.NEGATIVE["member"], negative_member)
        corpus = b"invented corpus bytes, not audio"
        ancestor = dict(C.ANCESTOR, zip_bytes=len(state_raw), zip_sha256=C.core.sha256(state_raw))
        negative = dict(C.NEGATIVE, zip_bytes=len(negative_raw), zip_sha256=C.core.sha256(negative_raw),
                        member_sha256=C.core.sha256(negative_member))
        source = dict(C.CORPUS, bytes=len(corpus), sha256=C.core.sha256(corpus))
        runs = [{"status": "completed", "conclusion": "success", "run_attempt": 1,
                 "head_sha": C.ANCESTOR["data_commit"], "event": "push"},
                {"status": "completed", "conclusion": "success", "run_attempt": 1,
                 "head_sha": C.NEGATIVE_HEAD, "event": "pull_request"}]
        metadata = [{"size_in_bytes": len(state_raw), "workflow_run": {
                         "id": C.ANCESTOR["run_id"], "head_sha": C.ANCESTOR["data_commit"]}},
                    {"size_in_bytes": len(negative_raw), "workflow_run": {
                         "id": C.NEGATIVE["run_id"], "head_sha": C.NEGATIVE_HEAD}}]
        for corrupt in (False, True):
            state = self.root / ("corrupt-prepare" if corrupt else "good-prepare")
            with self.subTest(corrupt=corrupt), self.environment(), \
                 mock.patch.object(C, "ANCESTOR", ancestor), mock.patch.object(C, "NEGATIVE", negative), \
                 mock.patch.object(C, "CORPUS", source), mock.patch.object(C.pool, "validate_ancestor"), \
                 mock.patch.object(C.core, "named_artifacts", return_value=[]), \
                 mock.patch.object(C.core, "github_get", side_effect=runs), \
                 mock.patch.object(C.core, "artifact_metadata", side_effect=metadata), \
                 mock.patch.object(C.core, "artifact_bytes", side_effect=[state_raw, negative_raw + b"x" if corrupt else negative_raw]), \
                 mock.patch.object(C, "download_corpus", side_effect=lambda path: path.write_bytes(corpus)) as transfer, \
                 mock.patch.object(C.core, "output") as output, \
                 mock.patch.object(C.native, "run_bounded", side_effect=AssertionError("prepare cannot execute models")):
                if corrupt:
                    with self.assertRaisesRegex(ValueError, "size/hash"):
                        C.prepare(self.path, state)
                    self.assertFalse((state / "claim/CLAIM.json").exists())
                    transfer.assert_not_called()
                else:
                    C.prepare(self.path, state)
                    self.assertTrue((state / "claim/CLAIM.json").is_file())
                    self.assertTrue(output.call_args.kwargs["execute"])
                    transfer.assert_called_once()

    def test_successful_execute_uses_original_inputs_once_then_saved_only_verification(self):
        state, pipeline, script = self.execution_fixture()
        with self.environment(), mock.patch.object(C.subprocess, "check_output", side_effect=[C.PIPELINE_COMMIT, "", script.read_bytes()]), \
             mock.patch.object(C, "validate_checkout_sources"), mock.patch.object(C, "validate_inputs") as validation, \
             mock.patch.object(C.native, "run_bounded", return_value=0) as child:
            C.execute(self.path, state, pipeline, Path("/invented/python"))
        self.assertEqual(validation.call_count, 2)
        self.assertEqual(child.call_count, 2)
        first, saved = child.call_args_list
        self.assertIn("--source-tar", first.args[0])
        self.assertIn("--negative-evidence-dir", first.args[0])
        self.assertNotIn("--source-tar", saved.args[0])
        self.assertNotIn("--negative-evidence-dir", saved.args[0])
        self.assertEqual(first.kwargs["seconds"], 900)
        self.assertEqual(saved.kwargs["seconds"], 180)
        self.assertNotIn("GH_TOKEN", first.kwargs["environment"])
        C.execution_ok(C.core.read_json(state / "result/EXECUTION.json"), self.digest)

    def test_failed_execute_retains_failure_and_never_starts_saved_verifier(self):
        state, pipeline, script = self.execution_fixture()
        with self.environment(), mock.patch.object(C.subprocess, "check_output", side_effect=[C.PIPELINE_COMMIT, "", script.read_bytes()]), \
             mock.patch.object(C, "validate_checkout_sources"), mock.patch.object(C, "validate_inputs"), \
             mock.patch.object(C.native, "run_bounded", return_value=1) as child:
            with self.assertRaises(ValueError):
                C.execute(self.path, state, pipeline, Path("/invented/python"))
        child.assert_called_once()
        self.assertEqual(C.core.read_json(state / "result/EXECUTION.json")["execute_exit_code"], 1)
        with self.environment():
            C.seal(self.path, state)
        terminal = C.core.read_json(state / "result/TERMINAL.json")
        self.assertEqual(terminal["status"], "FAILED")
        with self.assertRaisesRegex(ValueError, "stays consumed"):
            C.validate_terminal(state / "result", self.request, self.digest)

    def test_execution_never_accepts_boolean_float_or_partial_exit_codes(self):
        value = {"schema": "kws-research-v4-native-c-librispeech-execution-v1", "request_sha256": self.digest,
                 "execute_exit_code": 0, "verify_exit_code": 0, "optimizer_updates": 0,
                 "source_inputs_before_after_verified": True}
        C.execution_ok(value, self.digest)
        for key, changed in (("execute_exit_code", False), ("verify_exit_code", 0.0), ("verify_exit_code", None),
                             ("optimizer_updates", False), ("source_inputs_before_after_verified", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.execution_ok(dict(value, **{key: changed}), self.digest)

    def test_dispatch_source_contract_still_requires_one_added_immutable_request(self):
        directory = C.core.REQUEST_DIR + "/"
        path = directory + C.REQUEST_ID + ".json"
        self.assertEqual(C.core.changed_request("A\t" + path, False), path)
        for lines in ("M\t" + path, "D\t" + path, "A\t" + path + "\nA\t" + directory + "another-v1.json"):
            with self.assertRaises(ValueError):
                C.core.changed_request(lines, False)
        consumer = C.core.native_consumer(self.path)
        self.assertEqual(consumer.EXPERIMENT, C.EXPERIMENT)

    def source_fixture(self):
        names = [C.ENTRYPOINT] + ["research/fixture/reader" + str(index) + ".py" for index in range(9)]
        files = {name: b"# invented allowlist fixture; never executed\n" for name in names}
        allowed = {name: C.core.sha256(raw) for name, raw in files.items()}
        return files, allowed

    @staticmethod
    def source_zip(path, files):
        with zipfile.ZipFile(path, "w") as archive:
            for name, raw in files.items():
                archive.writestr(name, raw)

    def test_reviewed_source_sha_prevents_marker_execution_despite_rehashed_zip(self):
        files, allowed = self.source_fixture()
        archive = self.root / "SOURCE.zip"
        marker = self.root / "must-not-exist"
        self.source_zip(archive, files)
        with mock.patch.dict(C.SOURCE_SHA256, allowed, clear=True):
            self.assertEqual(C.verified_source(archive), files)
            files[C.ENTRYPOINT] = ("from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n").encode()
            self.source_zip(archive, files)
            rehashed_zip = C.file_sha256(archive)
            self.assertTrue(C.core.valid_sha(rehashed_zip, 64))
            with mock.patch.object(C.subprocess, "run", side_effect=AssertionError("marker must not execute")) as child:
                with self.assertRaisesRegex(ValueError, "reviewed allowlist"):
                    C.verified_source(archive)
            child.assert_not_called()
        self.assertFalse(marker.exists())

    def test_reviewed_source_rejects_duplicate_extra_missing_symlink_and_traversal(self):
        files, allowed = self.source_fixture()
        archive = self.root / "SOURCE.zip"
        for mutation in ("duplicate", "extra", "missing", "symlink", "traversal"):
            with self.subTest(mutation=mutation), mock.patch.dict(C.SOURCE_SHA256, allowed, clear=True):
                selected = dict(files)
                if mutation == "missing":
                    selected.pop(C.ENTRYPOINT)
                if mutation in ("extra", "traversal"):
                    selected["../escape.py" if mutation == "traversal" else "extra.py"] = b"x"
                with zipfile.ZipFile(archive, "w") as out:
                    for name, raw in selected.items():
                        if mutation == "symlink" and name == C.ENTRYPOINT:
                            info = zipfile.ZipInfo(name)
                            info.external_attr = 0o120777 << 16
                            out.writestr(info, raw)
                        else:
                            out.writestr(name, raw)
                    if mutation == "duplicate":
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            out.writestr(C.ENTRYPOINT, files[C.ENTRYPOINT])
                with self.assertRaisesRegex(ValueError, "path closure"):
                    C.verified_source(archive)

    def test_actual_checkout_must_match_the_reviewed_source_allowlist(self):
        files, allowed = self.source_fixture()
        tree = self.root / "source"
        for name, raw in files.items():
            path = tree / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        with mock.patch.dict(C.SOURCE_SHA256, allowed, clear=True):
            C.validate_checkout_sources(tree)
            (tree / C.ENTRYPOINT).write_text("raise RuntimeError('different source')\n")
            with self.assertRaisesRegex(ValueError, "reviewed source"):
                C.validate_checkout_sources(tree)

    def test_reviewed_source_roster_is_complete_and_the_fixed_consumer_is_not_generic(self):
        self.assertEqual(len(C.SOURCE_SHA256), 10)
        self.assertIn(C.ENTRYPOINT, C.SOURCE_SHA256)
        self.assertTrue(all(C.core.valid_sha(value, 64) for value in C.SOURCE_SHA256.values()))
        self.assertEqual(C.FIXED["max_execution_seconds"], 900)
        self.assertEqual(C.FIXED["profile"]["full_passes"], 2)

    def test_saved_verifier_only_gets_fixed_read_mode_stdlib_and_credential_free_environment(self):
        files, _ = self.source_fixture()
        root = self.root / "result"
        root.mkdir()
        output = {"schema": "invented-saved-receipt", "model_forward_calls": 0}
        process = subprocess.CompletedProcess([], 0, json.dumps(output), "")
        with mock.patch.dict(os.environ, {"GH_TOKEN": "invented-secret", "GITHUB_ACTIONS": "true",
                                         "GITHUB_RUN_ID": "999", "AWS_ACCESS_KEY_ID": "invented-secret"}), \
             mock.patch.object(C.subprocess, "run", return_value=process) as child:
            self.assertEqual(C.run_saved_verifier(root, files), output)
        child.assert_called_once()
        command = child.call_args.args[0]
        self.assertEqual(command[1:4], ["-I", "-S", "-B"])
        self.assertEqual(command[-2:], ["--verify-result-dir", str(root.resolve())])
        self.assertEqual(len(command), 7)
        self.assertEqual(child.call_args.kwargs["timeout"], 180)
        environment = child.call_args.kwargs["env"]
        for name in ("GH_TOKEN", "GITHUB_ACTIONS", "GITHUB_RUN_ID", "AWS_ACCESS_KEY_ID"):
            self.assertNotIn(name, environment)
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")

    def test_saved_verifier_failure_and_timeout_cannot_be_success(self):
        files, _ = self.source_fixture()
        root = self.root / "result"
        root.mkdir()
        for process in (subprocess.CompletedProcess([], 1, "{}", "error"),
                        subprocess.CompletedProcess([], 0, "{}", "unexpected warning"),
                        subprocess.CompletedProcess([], 0, "x" * 65536, "")):
            with mock.patch.object(C.subprocess, "run", return_value=process), self.assertRaises(ValueError):
                C.run_saved_verifier(root, files)
        with mock.patch.object(C.subprocess, "run", side_effect=subprocess.TimeoutExpired("saved", 180)) as child:
            with self.assertRaises(subprocess.TimeoutExpired):
                C.run_saved_verifier(root, files)
        child.assert_called_once()


if __name__ == "__main__":
    unittest.main()
