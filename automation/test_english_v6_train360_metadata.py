"""Pure stdlib fixtures: no archive download, source, audio, or model execution."""
from __future__ import annotations

import ast
import copy
from email.message import Message
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock


PATH = Path(__file__).with_name("english_v6_train360_metadata.py")
MODULE_SPEC = importlib.util.spec_from_file_location("train360_metadata", PATH)
m = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(m)


def fixture(extra=(), omit=(), replacement=None):
    census = b"; original fixture census\n1 | F | train-clean-360 | 1 | ONE\n2 | M | train-clean-360 | 1 | TWO\n"
    entries = [
        ("LibriSpeech", b"", tarfile.DIRTYPE),
        ("LibriSpeech/train-clean-360", b"", tarfile.DIRTYPE),
        *[(name, census if name.endswith("SPEAKERS.TXT") else (name + "\n").encode(), tarfile.REGTYPE)
          for name in sorted(m.ROOT_TEXT)],
        (m.PREFIX + "1/10/1-10.trans.txt", b"1-10-0001 HEY\n1-10-0002 THIS TEXT, IS PRESERVED.\n", tarfile.REGTYPE),
        (m.PREFIX + "2/20/2-20.trans.txt", b"2-20-0001 A B C D\n2-20-0002 ONE TWO THREE FOUR\n", tarfile.REGTYPE),
    ]
    for speaker, chapter in ((1, 10), (2, 20)):
        for utterance in (1, 2):
            name = m.PREFIX + f"{speaker}/{chapter}/{speaker}-{chapter}-{utterance:04d}.flac"
            entries.append((name, b"NOT-A-DECODABLE-AUDIO-FILE\x00" + name.encode(), tarfile.REGTYPE))
    entries.extend(extra)
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode="w", format=tarfile.USTAR_FORMAT) as out:
        for name, raw, kind in entries:
            if name in omit:
                continue
            if replacement is not None and name in replacement:
                raw = replacement[name]
            item = tarfile.TarInfo(name)
            item.type = kind
            item.size = len(raw)
            out.addfile(item, io.BytesIO(raw))
    payload = gzip.compress(tar.getvalue(), mtime=0)
    spec = copy.deepcopy(m.registration())
    spec["corpus"].update(bytes=len(payload), sha256=m.digest(payload), md5=hashlib.md5(payload).hexdigest())
    spec["completeness"].update(official_speakers_sha256=m.digest(census), split_census_readers=2)
    return payload, spec, census


def repin(payload, spec):
    spec = copy.deepcopy(spec)
    spec["corpus"].update(bytes=len(payload), sha256=m.digest(payload), md5=hashlib.md5(payload).hexdigest())
    return spec


def all_census_transport_fixture():
    """All existing census rows, no sampling; made-up text and non-audio bytes."""
    spec = m.registration()
    census = (m.ROOT / m.CENSUS).read_bytes()
    readers = m.census_ids(census, spec)
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode="w", format=tarfile.GNU_FORMAT) as out:
        def add(name, raw):
            item = tarfile.TarInfo(name)
            item.size = len(raw)
            out.addfile(item, io.BytesIO(raw))
        for name in sorted(m.ROOT_TEXT):
            add(name, census if name.endswith("SPEAKERS.TXT") else b"FIXTURE ONLY\n")
        for speaker in sorted(readers):
            stem = m.PREFIX + speaker + "/1/" + speaker + "-1"
            add(stem + ".trans.txt", (speaker + "-1-0000 FIXTURE ONLY\n").encode())
            add(stem + "-0000.flac", b"OPAQUE FIXTURE, NOT AUDIO")
    payload = gzip.compress(tar.getvalue(), mtime=0)
    return payload, repin(payload, spec)


class MetadataContracts(unittest.TestCase):
    def test_complete_original_text_and_opaque_flac_identities(self):
        raw, spec, census = fixture()
        metrics = {}
        catalog, index, report = m.scan_stream(io.BytesIO(raw), spec, census, operational=metrics)
        self.assertEqual(report["census_readers"], 2)
        self.assertEqual(report["utterances"], 4)
        self.assertEqual(report["compressed_bytes"], len(raw))
        self.assertEqual(report["compressed_sha256"], m.digest(raw))
        self.assertTrue(report["gzip_crc_and_trailer_verified"])
        self.assertIsNone(catalog["selected_people"])
        self.assertFalse(catalog["eligibility_computed"])
        self.assertIn("HEY\n", catalog["transcripts"][0]["text"])
        self.assertIn("TEXT,", catalog["transcripts"][0]["text"])
        flacs = [x for x in index["members"] if x["kind"] == "opaque_flac"]
        self.assertEqual(len(flacs), 4)
        for item in flacs:
            original = b"NOT-A-DECODABLE-AUDIO-FILE\x00" + item["path"].encode()
            self.assertEqual((item["bytes"], item["sha256"]), (len(original), m.digest(original)))
        self.assertEqual(metrics["opaque_flac_payload_bytes_hashed"], sum(x["bytes"] for x in flacs))
        self.assertEqual(metrics["status"], "COMPLETE_AUTHENTICATED_METADATA")

    def test_full_compressed_identity_and_eof(self):
        raw, spec, census = fixture()
        variants = []
        for field in ("sha256", "md5"):
            changed = copy.deepcopy(spec)
            changed["corpus"][field] = "0" * len(changed["corpus"][field])
            variants.append((raw, changed))
        longer = copy.deepcopy(spec)
        longer["corpus"]["bytes"] += 1
        variants.extend([(raw, longer), (raw[:-4], repin(raw[:-4], spec)), (raw + b"X", repin(raw + b"X", spec))])
        broken_crc = raw[:-8] + bytes([raw[-8] ^ 1]) + raw[-7:]
        variants.append((broken_crc, repin(broken_crc, spec)))
        for payload, changed in variants:
            with self.subTest(payload=len(payload), changed=changed["corpus"]), self.assertRaises(Exception):
                m.scan_stream(io.BytesIO(payload), changed, census)

    def test_tar_end_and_trailing_nonzero_bytes(self):
        raw, spec, census = fixture()
        tar = gzip.decompress(raw)
        _, _, report = m.scan_stream(io.BytesIO(raw), spec, census)
        end = report["tar_end_blocks_offset"]
        variants = [tar[:end], tar[:end + 512], tar[:end + 1024] + b"X" + bytes(511)]
        for changed in variants:
            payload = gzip.compress(changed, mtime=0)
            with self.subTest(length=len(changed)), self.assertRaises(RuntimeError):
                m.scan_stream(io.BytesIO(payload), repin(payload, spec), census)

    def test_duplicate_missing_extra_and_census_are_rejected(self):
        flac = m.PREFIX + "1/10/1-10-0001.flac"
        trans = m.PREFIX + "1/10/1-10.trans.txt"
        cases = [dict(extra=[(flac, b"X", tarfile.REGTYPE)]), dict(omit=[flac]),
                 dict(extra=[(m.PREFIX + "1/10/1-10-9999.flac", b"X", tarfile.REGTYPE)]),
                 dict(omit=[trans]), dict(replacement={trans: b"1-10-0001 A\n1-10-0001 B\n"}),
                 dict(replacement={trans: b"1-11-0001 A\n1-10-0002 B\n"}),
                 dict(replacement={"LibriSpeech/SPEAKERS.TXT": b"other census\n"})]
        for change in cases:
            raw, spec, census = fixture(**change)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                m.scan_stream(io.BytesIO(raw), spec, census)

    def test_other_splits_paths_and_extended_members_are_rejected(self):
        cases = [("LibriSpeech/dev-clean/1/10/1-10-0001.flac", b"X", tarfile.REGTYPE),
                 ("LibriSpeech/../other", b"X", tarfile.REGTYPE),
                 ("/LibriSpeech/train-clean-360", b"", tarfile.DIRTYPE),
                 ("LibriSpeech/UNKNOWN.TXT", b"X", tarfile.REGTYPE),
                 (m.PREFIX + "1/10/link", b"", tarfile.SYMTYPE),
                 (m.PREFIX + "1/10/hardlink", b"", tarfile.LNKTYPE),
                 ("pax", b"1 X\n", tarfile.XHDTYPE),
                 ("long-name", b"name\0", tarfile.GNUTYPE_LONGNAME)]
        for item in cases:
            raw, spec, census = fixture(extra=[item])
            with self.subTest(item=item), self.assertRaises(RuntimeError):
                m.scan_stream(io.BytesIO(raw), spec, census)

    def test_capacity_and_deadline_fail_closed(self):
        raw, spec, census = fixture()
        for key, value in (("expanded_tar_bytes", 511), ("archive_members", 1), ("all_text_bytes", 1),
                           ("single_text_bytes", 1), ("single_flac_bytes", 1), ("member_path_bytes", 3),
                           ("transcript_line_bytes", 1)):
            changed = copy.deepcopy(spec)
            changed["limits"][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                m.scan_stream(io.BytesIO(raw), changed, census)
        clock = iter((0, 20000))
        with self.assertRaises(RuntimeError):
            m.scan_stream(io.BytesIO(raw), spec, census, clock=lambda: next(clock))

    def test_failure_retains_partial_operational_identity(self):
        raw, spec, census = fixture(omit=[m.PREFIX + "1/10/1-10-0001.flac"])
        observed = {}
        with self.assertRaises(RuntimeError):
            m.scan_stream(io.BytesIO(raw), spec, census, operational=observed)
        self.assertEqual(observed["status"], "INCOMPLETE")
        self.assertEqual(observed["compressed_bytes"], len(raw))
        self.assertEqual(observed["compressed_sha256_observed"], m.digest(raw))
        self.assertGreater(observed["opaque_flac_payload_bytes_hashed"], 0)

    def test_retained_text_member_and_offsets_are_bound(self):
        raw, spec, census = fixture()
        catalog, index, _ = m.scan_stream(io.BytesIO(raw), spec, census)
        changed = copy.deepcopy(catalog)
        changed["transcripts"][0]["text"] += "X"
        with self.assertRaises(RuntimeError):
            m.validate_catalog(changed, index, spec, census)
        changed = copy.deepcopy(index)
        changed["members"][0]["tar_header_offset"] += 512
        with self.assertRaises(RuntimeError):
            m.validate_catalog(catalog, changed, spec, census)
        changed = copy.deepcopy(index)
        changed["members"][0]["sha256"] = "0" * 64
        # A directory has no payload and must hash as the empty byte sequence.
        with self.assertRaises(RuntimeError):
            m.validate_catalog(catalog, changed, spec, census)

    def test_bounded_output_never_publishes_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            with self.assertRaises(RuntimeError):
                m.record(path, {"text": "X" * 100}, maximum=16)
            self.assertFalse(path.exists())
            self.assertTrue(path.with_name(path.name + ".INCOMPLETE").exists())

    def test_http_origin_length_encoding_and_redirects_are_fixed(self):
        spec = m.registration()
        def response(**changes):
            class R:
                status = 200
                url = spec["corpus"]["url"]
            r = R()
            r.headers = Message()
            r.headers["Content-Length"] = str(spec["corpus"]["bytes"])
            for key, value in changes.items():
                if key in ("status", "url"):
                    setattr(r, key, value)
                else:
                    r.headers[key] = value
            return r
        m.validate_response(response(), spec)
        for change in ({"status": 206}, {"url": "https://other.invalid/file"}, {"Content-Range": "bytes 0-1/2"},
                       {"Content-Encoding": "gzip"}, {"Transfer-Encoding": "chunked"}, {"Content-Length": "1"}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                m.validate_response(response(**change), spec)
        request = m.urllib.request.Request(spec["corpus"]["url"])
        with self.assertRaises(m.urllib.error.HTTPError):
            m.NoRedirect().redirect_request(request, None, 302, "redirect", {}, spec["corpus"]["url"])

    def _failed_http_collection(self, open_response, expected_error):
        # These transport-only fixtures mock the already-covered capture/entry
        # boundary. The opener and scan boundary cannot access a real network.
        _, spec, _ = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts = root / "receipts"
            receipts.mkdir()
            execution = dict(run_id="123", run_head="2" * 40, run_branch=m.BRANCH,
                             run_attempt=1, prepared_parent="1" * 40)
            calls = []
            class Opener:
                def open(inner_self, request, timeout):
                    calls.append(request.full_url)
                    attempt = json.loads((receipts / "GET_ATTEMPT.json").read_bytes())
                    self.assertEqual(attempt["attempt"], 1)
                    self.assertEqual(attempt["url"], spec["corpus"]["url"])
                    self.assertEqual(attempt["method"], "GET")
                    self.assertEqual(request.get_header("Accept-encoding"), "identity")
                    self.assertEqual(timeout, spec["transport"]["socket_timeout_seconds"])
                    return open_response(request)
            with mock.patch.object(m, "registration", return_value=spec), \
                 mock.patch.object(m, "identity", return_value=execution), \
                 mock.patch.object(m, "validate_capture"), \
                 mock.patch.object(m, "once", return_value={"fixture_only": True}), \
                 mock.patch.object(m.resource, "setrlimit"), \
                 mock.patch.object(m.signal, "setitimer"), \
                 mock.patch.object(m.urllib.request, "build_opener", return_value=Opener()), \
                 mock.patch.object(m, "scan_stream", side_effect=AssertionError("body must not be read")) as scan:
                with self.assertRaises(expected_error):
                    m.collect(root, execution["prepared_parent"])
            self.assertEqual(calls, [spec["corpus"]["url"]])
            scan.assert_not_called()
            self.assertFalse((receipts / "COMPLETE").exists())
            result = json.loads((receipts / "RESULT.json").read_bytes())
            self.assertEqual(result["status"], "METADATA_ACQUISITION_INCOMPLETE")
            self.assertEqual((result["archive_get_attempted"], result["archive_response_returned"]), (1, 1))
            self.assertEqual(result["transport_observation"]["compressed_bytes"], 0)
            response = json.loads((receipts / "HTTP_RESPONSE.json").read_bytes())
            self.assertFalse(response["body_saved"])
            self.assertEqual((response["redirects"], response["retries"]), (0, 0))
            return response

    def test_invalid_http_headers_are_retained_before_rejection_without_body_read(self):
        class Response(io.BytesIO):
            status = 200
            url = m.registration()["corpus"]["url"]
            headers = Message()
            def read(self, *args):
                raise AssertionError("invalid response body must not be read")
        Response.headers["Content-Length"] = "1"
        Response.headers["X-Fixture"] = "original-invalid-length"
        body = Response(b"UNREAD FIXTURE")
        saved = self._failed_http_collection(lambda request: body, RuntimeError)
        self.assertEqual(saved["status"], 200)
        self.assertIn(["Content-Length", "1"], saved["headers"])
        self.assertIn(["X-Fixture", "original-invalid-length"], saved["headers"])
        self.assertFalse(saved["redirect_rejected"])
        self.assertTrue(body.closed)

    def test_http_error_and_redirect_rejection_retain_response_without_retry(self):
        class UnreadBody(io.BytesIO):
            def read(self, *args):
                raise AssertionError("error or redirect response body must not be read")
        for status in (403, 302):
            with self.subTest(status=status):
                headers = Message()
                headers["X-Fixture"] = "original-error-headers"
                if status == 302:
                    headers["Location"] = "https://other.invalid/not-followed"
                body = UnreadBody(b"UNREAD FIXTURE")
                def fail(request):
                    if status == 302:
                        return m.NoRedirect().redirect_request(request, body, status, "fixture redirect",
                            headers, headers["Location"])
                    raise m.urllib.error.HTTPError(request.full_url, status, "fixture forbidden", headers, body)
                saved = self._failed_http_collection(fail, m.urllib.error.HTTPError)
                self.assertEqual(saved["status"], status)
                self.assertIn(["X-Fixture", "original-error-headers"], saved["headers"])
                self.assertEqual(saved["redirect_rejected"], status == 302)
                if status == 302:
                    self.assertIn(["Location", "https://other.invalid/not-followed"], saved["headers"])
                self.assertTrue(body.closed)

    def test_closed_captured_only_runtime_and_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            total = 0
            for name in m.paths():
                source = m.ROOT / name
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                total += target.stat().st_size
            self.assertLess(total, 1048576)
            output = subprocess.check_output([sys.executable, "-SIB", str(root / "automation/english_v6_train360_metadata.py"), "check"], cwd=root)
            self.assertEqual(json.loads(output)["source_or_model_calls"], 0)
            (root / m.REGISTRATION).unlink()
            result = subprocess.run([sys.executable, "-SIB", str(root / "automation/english_v6_train360_metadata.py"), "check"],
                                    cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertNotEqual(result.returncode, 0)

    def test_only_stdlib_and_no_audio_or_selection_imports(self):
        allowed = {"__future__", "argparse", "gzip", "hashlib", "json", "os", "pathlib", "re", "resource",
                   "shutil", "signal", "subprocess", "tarfile", "time", "urllib"}
        tree = ast.parse(PATH.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(item.name.split(".")[0] for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module.split(".")[0])
        self.assertLessEqual(imported, allowed)
        self.assertFalse(any(isinstance(n, ast.Assert) for n in ast.walk(tree)))
        self.assertNotIn("extractall", PATH.read_text())
        self.assertNotIn("extractfile", PATH.read_text())

    def test_workflow_only_activation_captures_every_runtime_resource(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            repo = parent / "repository"
            repo.mkdir()
            for name in m.paths():
                target = repo / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(m.ROOT / name, target)
            def git(*args):
                return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
                    "-c", "user.name=Metadata Fixture", "-c", "user.email=fixture@example.invalid", *args], cwd=repo)
            git("init", "-q")
            git("add", ".")
            git("commit", "-qm", "inert metadata fixture preparation")
            prepared = git("rev-parse", "HEAD").decode().strip()
            (repo / m.WORKFLOW).write_text((repo / m.TEMPLATE).read_text().replace(m.TOKEN, prepared))
            git("add", m.WORKFLOW)
            git("commit", "-qm", "fixture-only workflow activation")
            head = git("rev-parse", "HEAD").decode().strip()
            event = dict(before=prepared, after=head, ref=m.BRANCH, forced=False, deleted=False,
                         repository=dict(full_name=m.REPO, private=False))
            event_path = parent / "event.json"
            event_path.write_text(json.dumps(event))
            env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": m.REPO, "GITHUB_REF": m.BRANCH,
                   "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_RUN_ID": "123",
                   "GITHUB_SHA": head, "GITHUB_WORKSPACE": str(repo), "GITHUB_EVENT_PATH": str(event_path)}
            prepared_spec = importlib.util.spec_from_file_location("prepared_fixture", repo / "automation/english_v6_train360_metadata.py")
            prepared_module = importlib.util.module_from_spec(prepared_spec)
            prepared_spec.loader.exec_module(prepared_module)
            output = parent / "metadata"
            marker_uploaded = False
            calls = []
            original_check_output = subprocess.check_output
            run = dict(id=123, head_sha=head, head_branch=m.BRANCH.removeprefix("refs/heads/"),
                       run_attempt=1, event="push", repository=dict(full_name=m.REPO, private=False))
            def api_check_output(command, **kwargs):
                if command[0] != "gh":
                    return original_check_output(command, **kwargs)
                self.assertEqual(command[:2], ["gh", "api"])
                endpoint = command[2]
                calls.append(endpoint)
                if "/runs?" in endpoint:
                    value = dict(total_count=1, workflow_runs=[run])
                else:
                    self.assertIn("actions/artifacts?name=" + m.MARKER, endpoint)
                    artifact = dict(name=m.MARKER, expired=False, workflow_run=dict(id=123, head_sha=head))
                    value = dict(total_count=int(marker_uploaded), artifacts=[artifact] if marker_uploaded else [])
                return json.dumps(value).encode()
            with mock.patch.dict(prepared_module.os.environ, env), mock.patch.object(subprocess, "check_output", side_effect=api_check_output):
                authorization = prepared_module.prepare(str(output), prepared)
            captured_script = output / "producer/repository/automation/english_v6_train360_metadata.py"
            captured_spec = importlib.util.spec_from_file_location("captured_fixture", captured_script)
            captured_module = importlib.util.module_from_spec(captured_spec)
            captured_spec.loader.exec_module(captured_module)
            spec = captured_module.registration()
            captured_module.validate_capture(output, authorization["execution"], spec)
            proof = json.loads((output / "producer/PRODUCER.json").read_bytes())
            self.assertEqual([x["path"] for x in proof["files"]], m.paths())
            self.assertIn(m.REGISTRATION, [x["path"] for x in proof["files"]])
            self.assertFalse(authorization["scientific_panel_budget_consumed"])
            payload, transport_spec = all_census_transport_fixture()
            get_count = 0
            class Response(io.BytesIO):
                status = 200
                url = transport_spec["corpus"]["url"]
                headers = Message()
            Response.headers["Content-Length"] = str(len(payload))
            class Opener:
                def open(inner_self, request, timeout):
                    nonlocal get_count
                    get_count += 1
                    self.assertEqual(request.full_url, transport_spec["corpus"]["url"])
                    self.assertEqual(request.get_method(), "GET")
                    self.assertEqual(request.get_header("Accept-encoding"), "identity")
                    self.assertEqual(timeout, 60)
                    return Response(payload)
            marker_uploaded = True
            # Only compressed identity is substituted for this in-memory fixture;
            # the captured producer, real census, entry guards and file closure
            # remain exact. No network implementation can be called here.
            with mock.patch.dict(captured_module.os.environ, env), \
                 mock.patch.object(subprocess, "check_output", side_effect=api_check_output), \
                 mock.patch.object(captured_module, "registration", return_value=transport_spec), \
                 mock.patch.object(captured_module.urllib.request, "build_opener", return_value=Opener()), \
                 mock.patch.object(captured_module.resource, "setrlimit") as memory_limit, \
                 mock.patch.object(captured_module.signal, "setitimer"):
                result = captured_module.collect(str(output), prepared)
                validation = captured_module.verify_catalog(output / "receipts/COMPLETE")
            self.assertEqual(get_count, 1)
            self.assertEqual((result["archive_get_attempted"], result["archive_response_returned"]), (1, 1))
            self.assertEqual(len(calls), 4)
            self.assertEqual(result["status"], "COMPLETE_AUTHENTICATED_METADATA")
            self.assertEqual(validation["summary"]["census_readers"], 921)
            self.assertEqual(result["metadata_only_counters"], m.registration()["metadata_only_counters"])
            self.assertFalse(any(p.suffix == ".flac" for p in output.rglob("*")))
            memory_limit.assert_called_once_with(captured_module.resource.RLIMIT_AS,
                (transport_spec["limits"]["address_space_bytes"],) * 2)
            (output / "producer/repository" / m.REGISTRATION).unlink()
            with self.assertRaises(RuntimeError):
                captured_module.validate_capture(output, authorization["execution"], spec)

    def test_separate_first_push_and_all_failed_runs_count(self):
        parent, head = "1" * 40, "2" * 40
        env = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": m.REPO, "GITHUB_REF": m.BRANCH,
               "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_RUN_ID": "123", "GITHUB_SHA": head}
        event = dict(before=parent, after=head, ref=m.BRANCH, forced=False, deleted=False,
                     repository=dict(full_name=m.REPO, private=False))
        execution = m.identity(parent, env=env, event=event, parent=parent)
        for changes in ({"GITHUB_RUN_ATTEMPT": "2"}, {"GITHUB_REF": "refs/heads/old-source"}, {"GITHUB_EVENT_NAME": "workflow_dispatch"}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                m.identity(parent, env=dict(env, **changes), event=event, parent=parent)
        run = dict(id=123, head_sha=head, head_branch=m.BRANCH.removeprefix("refs/heads/"), run_attempt=1,
                   event="push", repository=dict(full_name=m.REPO, private=False))
        def api(endpoint):
            return dict(total_count=1, workflow_runs=[run]) if "/runs?" in endpoint else dict(total_count=0, artifacts=[])
        m.once(execution, False, api)
        def prior_failure(endpoint):
            return dict(total_count=2, workflow_runs=[run, dict(run, id=122, conclusion="failure")])
        with self.assertRaises(RuntimeError):
            m.once(execution, False, prior_failure)

    def test_only_registered_workflow_activation_can_replace_disabled_file(self):
        template = (m.ROOT / m.TEMPLATE).read_text()
        current = (m.ROOT / m.WORKFLOW).read_text()
        if "if: false" not in current:
            import re
            parents = re.findall(r"PREPARED_PARENT: '([0-9a-f]{40})'", current)
            self.assertEqual(len(parents), 1)
            self.assertEqual(current, template.replace(m.TOKEN, parents[0]))
        self.assertNotIn("workflow_dispatch", current)
        self.assertNotIn("workflow_dispatch", template)
        self.assertIn("producer/repository/automation/english_v6_train360_metadata.py\" collect", template)
        self.assertIn("timeout-minutes: 210", template)
        self.assertIn("timeout-minutes: 190", template)
        self.assertIn("/automation/english_v6_train360_metadata_registration.json", template)


if __name__ == "__main__":
    unittest.main()
