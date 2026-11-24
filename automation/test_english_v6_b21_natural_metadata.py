"""Offline adversarial tests of the independent text-only diagnostic boundary."""
from __future__ import annotations
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_natural_metadata as metadata


def metadata_fixture():
    assessment = copy.deepcopy(metadata.frozen())
    speakers = assessment["primary"]["natural"]["speakers"] + assessment["replication"]["natural"]["speakers"]
    census = "\n".join(speaker + " | F | train-clean-100 | 1 | Reader" for speaker in sorted(speakers)) + "\n"
    assessment["metadata_speakers_sha256"] = metadata.digest(census.encode())
    transcripts = []
    for speaker in speakers:
        content = speaker + "-1-0001 THIS IS ORIGINAL TEXT\n" + speaker + "-1-0002 THESE ARE OTHER WORDS\n"
        transcripts.append(dict(path="LibriSpeech/train-clean-100/" + speaker + "/1/" + speaker + "-1.trans.txt",
            text=content, sha256=metadata.digest(content.encode()), bytes=len(content.encode())))
    catalog = dict(speakers_text=census, speakers_sha256=assessment["metadata_speakers_sha256"],
        transcripts=sorted(transcripts, key=lambda row: row["path"]), corpus_sha256=metadata.CORPUS_SHA)
    return assessment, catalog


def replace_transcript(catalog, speaker, texts):
    item = next(row for row in catalog["transcripts"] if row["path"].split("/")[2] == speaker)
    item["text"] = "".join(speaker + "-1-" + str(i).zfill(4) + " " + text + "\n" for i, text in enumerate(texts, 1))
    item["sha256"] = metadata.digest(item["text"].encode())
    item["bytes"] = len(item["text"].encode())


def fake_archive(path):
    members = {
        "LibriSpeech/SPEAKERS.TXT": b"1 | F | train-clean-100 | 1 | Reader\n",
        "LibriSpeech/train-clean-100/1/2/1-2.trans.txt": b"1-2-0001 THESE ARE FOUR WORDS\n",
        "LibriSpeech/train-clean-100/1/2/1-2-0001.flac": b"TEST PLACEHOLDER; NOT AUDIO; MUST NEVER EXTRACT",
        "LibriSpeech/README.TXT": b"ignored metadata outside explicit allowlist",
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, raw in members.items():
            item = tarfile.TarInfo(name)
            item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
    return path.stat().st_size, metadata.digest(path.read_bytes())


class MetadataDiagnosticTests(unittest.TestCase):
    def test_frozen_primitives_do_not_import_source_or_audio(self):
        before = set(sys.modules)
        primitives = metadata.primitive_namespace()
        self.assertEqual(primitives["ranges"](81, 8), [dict(index=i, start=81*i//8, end=81*(i+1)//8-1) for i in range(8)])
        imported = set(sys.modules) - before
        self.assertFalse(any(name.startswith(("english_v6_b21_source", "english_v6_b21_fixed_mirror_source", "piper", "torch", "numpy", "soundfile")) for name in imported))
        self.assertNotIn("download", primitives)
        self.assertNotIn("lineage", primitives)

    def test_exact_original_predicate_edges(self):
        for panel in ("primary", "replication"):
            predicate = metadata.eligibility_predicate(panel)
            for words in (["WORD"]*4, ["WORD"]*14, ["DON'T", "CHANGE", "THIS", "TEXT"]):
                self.assertTrue(predicate(words))
            for words in (["WORD"]*3, ["WORD"]*15, ["HEY", "THREE", "OTHER", "WORDS"], ["WORD", "WORD", "WORD", "X-ONE"], ["Word"]*4, ["", "WORD", "WORD", "WORD"]):
                self.assertFalse(predicate(words))

    def test_text_scan_never_extracts_audio_member(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "corpus.tar.gz"
            size, sha = fake_archive(path)
            original = tarfile.TarFile.extractfile
            extracted = []
            def extract(archive, member):
                self.assertFalse(member.name.endswith(".flac"))
                extracted.append(member.name)
                return original(archive, member)
            with mock.patch.object(tarfile.TarFile, "extractfile", extract):
                catalog = metadata.scan_text_metadata(path, size, sha)
            self.assertEqual(extracted, ["LibriSpeech/SPEAKERS.TXT", "LibriSpeech/train-clean-100/1/2/1-2.trans.txt"])
            self.assertEqual(catalog["ignored_audio_members"], 1)
            self.assertEqual(catalog["audio_members_extracted"], 0)
            self.assertEqual(len(catalog["transcripts"]), 1)

    def test_wrong_whole_sha_stops_before_tar_open(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "corpus.tar.gz"
            size, _ = fake_archive(path)
            with mock.patch.object(tarfile, "open") as opener, self.assertRaisesRegex(RuntimeError, "SHA before tar"):
                metadata.scan_text_metadata(path, size, "0"*64)
            opener.assert_not_called()

    def test_all_original_readers_diagnosed_without_replacement(self):
        assessment, catalog = metadata_fixture()
        primary = assessment["primary"]["natural"]["speakers"]
        replica = assessment["replication"]["natural"]["speakers"]
        replace_transcript(catalog, primary[0], [" ".join(["LONG"]*15)]*2)
        replace_transcript(catalog, primary[1], ["THIS IS ORIGINAL TEXT", "ONLY THREE WORDS"])
        replace_transcript(catalog, replica[0], ["HEY THIS IS BLOCKED", "THIS HAS BAD-TOKEN WORDS"])
        result = metadata.diagnose(catalog, assessment)
        self.assertEqual(result["panels"]["primary"]["ineligible_readers"], primary[:2])
        self.assertEqual(result["panels"]["replication"]["ineligible_readers"], replica[:1])
        self.assertEqual(result["panels"]["primary"]["first_original_stop_reader"], primary[0])
        self.assertEqual(result["panels"]["primary"]["readers"][0]["rejection_reason_counts"], {"more_than_14_words": 2})
        for panel in ("primary", "replication"):
            self.assertEqual(result["panels"][panel]["original_readers"], assessment[panel]["natural"]["speakers"])
            self.assertEqual(len(result["panels"][panel]["readers"]), 12)
            self.assertIsNone(result["panels"][panel]["replacement_readers"])
            self.assertFalse(result["panels"][panel]["source_panel_opened"])
        self.assertEqual(result["original_failed_source_status"], "STOP")
        self.assertFalse(result["original_synthetic_regeneration_allowed"])

    def test_eligible_diagnostic_does_not_authorize_source(self):
        assessment, catalog = metadata_fixture()
        result = metadata.diagnose(catalog, assessment)
        for panel in result["panels"].values():
            self.assertEqual(panel["original_metadata_rank_status"], "ELIGIBLE_METADATA_ONLY")
            self.assertFalse(panel["source_panel_opened"])
            self.assertTrue(all(row["ranked_pair_status"].startswith("RETROSPECTIVE_METADATA_ONLY") for row in panel["readers"]))
        self.assertFalse(result["audio_or_source_continuation_authorized"])
        self.assertTrue(all(result[key] == 0 for key in metadata.ZERO))

    def test_partial_catalog_and_duplicate_transcripts_stop(self):
        assessment, catalog = metadata_fixture()
        catalog["transcripts"].pop()
        with self.assertRaisesRegex(RuntimeError, "full official"):
            metadata.diagnose(catalog, assessment)
        assessment, catalog = metadata_fixture()
        catalog["transcripts"].append(catalog["transcripts"][-1])
        with self.assertRaisesRegex(RuntimeError, "unique sorted"):
            metadata.diagnose(catalog, assessment)

    def test_guard_rejects_retry_private_and_wrong_parent(self):
        parent, head = "a"*40, "b"*40
        env = dict(GITHUB_ACTIONS="true", GITHUB_REPOSITORY=metadata.REPO, GITHUB_REF=metadata.BRANCH,
            GITHUB_EVENT_NAME="push", GITHUB_RUN_ATTEMPT="1", GITHUB_RUN_ID="123", GITHUB_SHA=head)
        event = dict(before=parent, after=head, ref=metadata.BRANCH, forced=False, deleted=False,
            repository=dict(full_name=metadata.REPO, private=False))
        self.assertEqual(metadata.execution_guard(parent, env, event, parent)["run_id"], "123")
        for bad_env, bad_event, bad_parent in ((dict(env, GITHUB_RUN_ATTEMPT="2"), event, parent),
                (env, dict(event, repository=dict(full_name=metadata.REPO, private=True)), parent), (env, event, "c"*40)):
            with self.assertRaises(RuntimeError):
                metadata.execution_guard(parent, bad_env, bad_event, bad_parent)

    def test_one_execution_rejects_second_run_and_existing_budget(self):
        identity = dict(run_id="123", run_head="b"*40, prepared_parent="a"*40, run_attempt=1, run_branch=metadata.BRANCH)
        run = dict(id=123, head_sha=identity["run_head"], head_branch=metadata.BRANCH.removeprefix("refs/heads/"),
            run_attempt=1, event="push", repository=dict(full_name=metadata.REPO, private=False))
        responses = [dict(total_count=1, workflow_runs=[run]), dict(total_count=0, artifacts=[])]
        metadata.one_execution(identity, False, mock.Mock(side_effect=responses))
        with self.assertRaises(RuntimeError):
            metadata.one_execution(identity, False, mock.Mock(return_value=dict(total_count=2, workflow_runs=[run, run])))
        with self.assertRaises(RuntimeError):
            metadata.one_execution(identity, False, mock.Mock(side_effect=[responses[0], dict(total_count=1, artifacts=[{}])]))

    def test_transport_joins_eight_ranges_and_refuses_resume(self):
        payload = bytes(range(128))*8
        primitives = metadata.primitive_namespace()
        spec = dict(metadata.transport_spec(), bytes=len(payload), sha256=metadata.digest(payload))
        requested = []
        class Response:
            status = 206
            def __init__(self, start, end):
                self.raw = io.BytesIO(payload[start:end+1])
                self.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}", "Content-Length": str(end-start+1)}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def geturl(self): return metadata.OFFICIAL[3]
            def read(self, count): return self.raw.read(count)
        class Opener:
            def open(self, request, timeout):
                start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
                requested.append((start, end))
                return Response(start, end)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus = root / "train-clean-100.tar.gz"
            receipts = root / "receipts"
            receipts.mkdir()
            with mock.patch.object(metadata.urllib.request, "build_opener", return_value=Opener()):
                report = metadata.download(corpus, receipts, spec, primitives)
                self.assertEqual(report["status"], "COMPLETE_VERIFIED")
                self.assertEqual(corpus.read_bytes(), payload)
                self.assertEqual(len(requested), 8)
                with self.assertRaisesRegex(RuntimeError, "no resume"):
                    metadata.download(corpus, receipts, spec, primitives)
            self.assertEqual(len(list((receipts / "TRANSPORT/ranges").glob("*-RETURNED.json"))), 8)

    def test_transport_wrong_complete_sha_retains_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipts = root / "receipts"
            receipts.mkdir()
            spec = dict(metadata.transport_spec(), bytes=8, sha256="0"*64)
            def transfer(part, spec, fd, receipts, stop, deadline, opener):
                os.pwrite(fd, b"X", part["start"])
                return dict(part, status="COMPLETE")
            primitives = dict(metadata.primitive_namespace(), transfer=transfer)
            with self.assertRaisesRegex(RuntimeError, "full original SHA"):
                metadata.download(root / "train-clean-100.tar.gz", receipts, spec, primitives)
            self.assertFalse((root / "train-clean-100.tar.gz").exists())
            self.assertEqual(json.loads((receipts / "TRANSPORT/RESULT.json").read_bytes())["status"], "STOP")

    def test_workflow_is_single_metadata_job_without_science_installs(self):
        raw = (metadata.ROOT / metadata.TEMPLATE).read_text()
        actual = (metadata.ROOT / metadata.WORKFLOW).read_text()
        binding = re.search(r"PREPARED_PARENT: '([^']+)'", actual).group(1)
        self.assertTrue(binding == metadata.TOKEN or re.fullmatch(r"[0-9a-f]{40}", binding) is not None)
        self.assertEqual(raw.replace(metadata.TOKEN, binding), actual)
        for prohibited in ("pip install", "piper-tts", "soundfile", "torch", "observe ", "generate ", "workflow_dispatch", "schedule:"):
            self.assertNotIn(prohibited, raw)
        self.assertEqual(raw.count("metadata:\n"), 1)
        self.assertLess(raw.index("name: english-v6-b21-natural-metadata-consumed"), raw.index("name: Verify one full corpus"))
        self.assertIn("include-hidden-files: true", raw)

    def test_real_git_activation_captures_exact_fourteen_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in set(metadata.FROZEN) | set(metadata.OWN):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(metadata.ROOT / name, target)
            (root / metadata.WORKFLOW).write_bytes((root / metadata.TEMPLATE).read_bytes())
            def git(*arguments):
                return subprocess.check_output(["git", "-c", "user.name=Offline Fixture", "-c", "user.email=fixture@example.invalid",
                    "-c", "commit.gpgsign=false", *arguments], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
            git("init", "-q")
            git("add", ".")
            git("commit", "-qm", "inert preparation")
            parent = git("rev-parse", "HEAD")
            (root / metadata.WORKFLOW).write_text((root / metadata.TEMPLATE).read_text().replace(metadata.TOKEN, parent))
            git("add", metadata.WORKFLOW)
            git("commit", "-qm", "workflow-only activation")
            identity = dict(run_id="123", run_head=git("rev-parse", "HEAD"), prepared_parent=parent,
                run_attempt=1, run_branch=metadata.BRANCH)
            files = metadata.verify_git(identity, root)
            self.assertEqual(len(files), 14)
            with (root / metadata.OWN[0]).open("ab") as stream:
                stream.write(b"\n# unauthenticated edit\n")
            with self.assertRaisesRegex(RuntimeError, "exact Git-bound"):
                metadata.verify_git(identity, root)


if __name__ == "__main__":
    unittest.main()
