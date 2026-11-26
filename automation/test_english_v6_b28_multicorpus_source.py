"""Invented tar containers only; no real audio, catalog draw, network, or model."""
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
import tarfile
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).with_name("english_v6_b28_multicorpus_source.py")
SPEC = importlib.util.spec_from_file_location("b28_multicorpus", SCRIPT)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def tar_bytes(entries):
    target = io.BytesIO()
    with tarfile.open(fileobj=target, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, raw, kind in entries:
            item = tarfile.TarInfo(name)
            item.type = kind
            item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
    return gzip.compress(target.getvalue(), mtime=0)


def fixture():
    census = ("; INVENTED FIXTURE, NOT LIBRISPEECH\n" + "".join(
        f"{i} | F | train-clean-{100 if i <= 6 else 360} | 1 | FIXTURE\n" for i in range(1, 13))).encode()
    corpora = copy.deepcopy(m.CORPORA)
    texts, indices, payloads, entries_all = {}, {}, {}, {}
    for split in corpora:
        readers = range(1, 7) if split == "train-clean-100" else range(7, 13)
        entries = [(name, census if name.endswith("SPEAKERS.TXT") else b"ORIGINAL ROOT FIXTURE\n", tarfile.REGTYPE)
                   for name in sorted(m.ROOT_TEXT)]
        transcripts, members = [], []
        for person in readers:
            stem = f"LibriSpeech/{split}/{person}/1/{person}-1"
            raw = (f"{person}-1-0001 ONE TWO THREE FOUR\n"
                   f"{person}-1-0002 PLEASE KEEP THIS SIMPLE\n"
                   f"{person}-1-0003 HEY THIS IS BLOCKED\n").encode()
            transcripts.append(dict(path=stem + ".trans.txt", bytes=len(raw), sha256=m.sha(raw), text=raw.decode()))
            entries.append((stem + ".trans.txt", raw, tarfile.REGTYPE))
            for n in range(1, 4):
                path = stem + f"-{n:04d}.flac"
                opaque = b"OPAQUE FIXTURE, NOT AUDIO: " + path.encode()
                entries.append((path, opaque, tarfile.REGTYPE))
                members.append(dict(path=path, kind="opaque_flac", bytes=len(opaque), sha256=m.sha(opaque)))
        payload = tar_bytes(entries)
        corpora[split].update(bytes=len(payload), sha256=m.sha(payload), md5=hashlib.md5(payload).hexdigest())
        transcripts.sort(key=lambda x: x["path"])
        if split == "train-clean-100":
            texts[split] = dict(schema="english-v6-b21-full-train-clean-100-text-metadata-v1",
                corpus_bytes=len(payload), corpus_sha256=m.sha(payload), speakers_text=census.decode(),
                speakers_sha256=m.sha(census), metadata_only=True, transcripts=transcripts)
            indices[split] = None
        else:
            corpus = m.corpus_without_split(corpora[split])
            roots = [dict(path=n, bytes=len(r), sha256=m.sha(r), text=r.decode()) for n, r, _ in entries if n in m.ROOT_TEXT]
            texts[split] = dict(schema="english-v6-train360-complete-text-catalog-v1", split=split,
                corpus=corpus, eligibility_computed=False, selected_people=None, root_metadata=roots, transcripts=transcripts)
            indices[split] = dict(schema="english-v6-train360-complete-member-index-v1", split=split, corpus=corpus, members=members)
        payloads[split], entries_all[split] = payload, entries
    return census, corpora, texts, indices, payloads, entries_all


class Response(io.BytesIO):
    def __init__(self, raw, corpus):
        super().__init__(raw)
        self.status, self.url = 200, corpus["url"]
        self.headers = Message()
        self.headers["Content-Length"] = str(corpus["bytes"])


class MultiCorpusContracts(unittest.TestCase):
    def setUp(self):
        self.census, self.corpora, self.texts, self.indices, self.payloads, self.entries = fixture()
        self.patch_corpora = mock.patch.object(m, "CORPORA", self.corpora)
        self.patch_census = mock.patch.object(m, "SPEAKERS_SHA", m.sha(self.census))
        self.patch_corpora.start(); self.patch_census.start()
        self.addCleanup(self.patch_corpora.stop); self.addCleanup(self.patch_census.stop)
        self.catalogs = {k: m.normalized_catalog(self.texts[k], k, self.census, self.indices[k]) for k in self.corpora}
        self.rows = [r for cat in self.catalogs.values() for r in cat["rows"]]
        self.people, self.roster = m.select_natural_rows(self.rows, [], "fixture-salt-before-any-audio")
        self.plan = dict(schema="english-v6-b28-multicorpus-staging-plan-v1", panel="primary",
            assessment_sha256="1" * 64, training_readback_sha256="2" * 64, roster=self.roster,
            corpora=self.corpora, catalog_pins={k: dict(bytes=len(m.encoded(c)), sha256=m.sha(m.encoded(c))) for k, c in self.catalogs.items()},
            limits=m.LIMITS)

    def opener(self, calls):
        def open_response(corpus):
            calls.append(corpus["split"])
            return Response(self.payloads[corpus["split"]], corpus)
        return open_response

    def test_original_metadata_bijection_and_unchanged_word_rule(self):
        self.assertEqual(len(self.people), 12)
        self.assertEqual(len(self.roster), 24)
        self.assertFalse(any("HEY" in r["transcript"] for r in self.rows))
        for text in ["One TWO THREE FOUR", "ONE  TWO THREE FOUR", "ONE TWO Mero FOUR", "HEY ONE TWO THREE"]:
            self.assertFalse(m.text_eligible(text))
        self.assertTrue(m.text_eligible("IT'S A NORMAL SENTENCE"))
        changed = copy.deepcopy(self.texts["train-clean-360"])
        changed["transcripts"][0]["text"] += "X"
        with self.assertRaises(ValueError):
            m.normalized_catalog(changed, "train-clean-360", self.census, self.indices["train-clean-360"])
        changed = copy.deepcopy(self.indices["train-clean-360"])
        changed["members"].pop()
        with self.assertRaises(ValueError):
            m.normalized_catalog(self.texts["train-clean-360"], "train-clean-360", self.census, changed)

    def test_cross_split_people_exclusion_and_no_redraw(self):
        with self.assertRaises(ValueError):
            m.select_natural_rows(self.rows, [self.people[0]], "fixture-salt-before-any-audio")
        extra = copy.deepcopy(self.rows[0]); extra["corpus_split"] = extra["archive_key"] = "train-clean-360"
        extra["source_id"] += "9"
        with self.assertRaises(ValueError):
            m.select_natural_rows(self.rows + [extra], [], "fixture-salt-before-any-audio")
        with self.assertRaises(ValueError):
            m.select_natural_rows(self.rows + [self.rows[0]], [], "fixture-salt-before-any-audio")

    def test_two_complete_streams_release_exactly24_opaque_originals(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source-stage"
            release = m.stage_panel(self.plan, root, self.catalogs, self.opener(calls))
            actual = m.read_release(root / "COMPLETE", self.plan, self.catalogs)
            self.assertEqual(calls, ["train-clean-100", "train-clean-360"])
            self.assertEqual(len(actual), 24)
            self.assertEqual([x["row"] for x in actual], self.roster)
            self.assertEqual(len(list((root / "COMPLETE").rglob("*.flac"))), 24)
            self.assertEqual(release["native_decodes"], 0)
            self.assertFalse(release["original_compressed_archives_rehashed_offline"])
            self.assertEqual(sum(x["compressed_bytes"] for x in release["corpus_receipts"].values()), sum(len(x) for x in self.payloads.values()))
            self.assertEqual({x["source_group_id"].split("-")[3] for x in actual}, {"100", "360"})
            for item in actual:
                self.assertTrue(item["id"].startswith("b28-primary-natural-"))
                self.assertTrue(item["path"].read_bytes().startswith(b"OPAQUE FIXTURE"))

    def test_second_stream_failure_never_releases_first_corpus(self):
        calls = []
        def fail_second(corpus):
            calls.append(corpus["split"])
            if corpus["split"] == "train-clean-360":
                raise OSError("invented second transport failure")
            return Response(self.payloads[corpus["split"]], corpus)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source-stage"
            with self.assertRaises(OSError):
                m.stage_panel(self.plan, root, self.catalogs, fail_second)
            self.assertFalse((root / "COMPLETE").exists())
            self.assertTrue((root / "INCOMPLETE/train-clean-100").is_dir())
            result = json.loads((root / "RESULT.json").read_bytes())
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertEqual(result["http_attempts"], calls)
            with self.assertRaises(ValueError):
                m.stage_panel(self.plan, root, self.catalogs, fail_second)
            self.assertEqual(len(calls), 2)

    def test_response_validation_happens_before_body_or_staging(self):
        for field, value in [("status", 206), ("url", "https://invalid.example/corpus"),
                              ("Content-Length", "1"), ("Content-Encoding", "gzip"), ("Transfer-Encoding", "chunked")]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                response = Response(self.payloads["train-clean-100"], self.corpora["train-clean-100"])
                if field in ("url", "status"):
                    setattr(response, field, value)
                else:
                    response.headers[field] = value
                response.read = mock.Mock(side_effect=AssertionError("response body was not permitted"))
                root = Path(tmp) / "stage"
                with self.assertRaises(ValueError):
                    m.stage_panel(self.plan, root, self.catalogs, lambda _: response)
                self.assertFalse((root / "COMPLETE").exists())
                response.read.assert_not_called()

    def test_full_eof_hash_crc_and_late_tar_tail_are_required(self):
        key = "train-clean-100"
        good = self.payloads[key]
        broken_crc = good[:-8] + bytes([good[-8] ^ 1]) + good[-7:]
        cases = [good[:-4], good + b"X", broken_crc, gzip.compress(gzip.decompress(good) + b"NONZERO", mtime=0)]
        for bad in cases:
            with self.subTest(length=len(bad)), tempfile.TemporaryDirectory() as tmp:
                corpus = dict(self.corpora[key], bytes=len(bad), sha256=m.sha(bad), md5=hashlib.md5(bad).hexdigest())
                progress = {}
                with self.assertRaises(Exception):
                    m.stage_archive(io.BytesIO(bad), corpus, [r for r in self.roster if r["archive_key"] == key],
                        self.catalogs[key], tmp, 1e100, progress=progress)
                self.assertGreater(progress["compressed_bytes"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            corpus = dict(self.corpora[key], sha256="0" * 64)
            with self.assertRaises(ValueError):
                m.stage_archive(io.BytesIO(good), corpus, [r for r in self.roster if r["archive_key"] == key], self.catalogs[key], tmp, 1e100)

    def test_missing_duplicate_foreign_or_nonordinary_member_stops(self):
        key = "train-clean-100"; entries = self.entries[key]
        chosen = next(r for r in self.roster if r["archive_key"] == key)["flac_member"]
        duplicate = next(x for x in entries if x[0] == chosen)
        cases = [[x for x in entries if x[0] != chosen], entries + [duplicate],
            entries + [("LibriSpeech/dev-clean/1/1/1-1-0001.flac", b"X", tarfile.REGTYPE)],
            entries + [("LibriSpeech/train-clean-100/1/1/link", b"", tarfile.SYMTYPE)],
            entries + [("LibriSpeech/../evil", b"X", tarfile.REGTYPE)]]
        for changed in cases:
            raw = tar_bytes(changed)
            corpus = dict(self.corpora[key], bytes=len(raw), sha256=m.sha(raw), md5=hashlib.md5(raw).hexdigest())
            with self.subTest(length=len(raw)), tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                m.stage_archive(io.BytesIO(raw), corpus, [r for r in self.roster if r["archive_key"] == key], self.catalogs[key], tmp, 1e100)

    def test_transcript_and_preexisting360_flac_identity_are_preserved(self):
        key = "train-clean-360"; row = next(r for r in self.roster if r["archive_key"] == key)
        for name in (row["flac_member"], row["transcript_member"]):
            changed = [(n, raw + b"X" if n == name else raw, typ) for n, raw, typ in self.entries[key]]
            raw = tar_bytes(changed)
            corpus = dict(self.corpora[key], bytes=len(raw), sha256=m.sha(raw), md5=hashlib.md5(raw).hexdigest())
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                m.stage_archive(io.BytesIO(raw), corpus, [r for r in self.roster if r["archive_key"] == key], self.catalogs[key], tmp, 1e100)

    def test_global_staging_budget_checked_before_next_member_write(self):
        key = "train-clean-100"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                m.stage_archive(io.BytesIO(self.payloads[key]), self.corpora[key],
                    [r for r in self.roster if r["archive_key"] == key], self.catalogs[key], tmp, 1e100, remaining_bytes=1)
            self.assertEqual(list(Path(tmp).iterdir()), [])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                m.stage_archive(io.BytesIO(self.payloads[key]), self.corpora[key],
                    [r for r in self.roster if r["archive_key"] == key], self.catalogs[key], tmp, -1)

    def test_plan_identity_corpus_and_row_mapping_fail_closed(self):
        for mutate in [lambda p: p["corpora"]["train-clean-360"].update(url="https://other.invalid/source"),
                       lambda p: p["roster"][0].update(flac_member="LibriSpeech/train-clean-100/1/1/not-original.flac"),
                       lambda p: p["catalog_pins"]["train-clean-100"].update(sha256="0" * 64),
                       lambda p: p["roster"].pop()]:
            altered = copy.deepcopy(self.plan); mutate(altered)
            with self.assertRaises(ValueError):
                m.validate_plan(altered, self.catalogs)

    def test_released_file_tamper_and_extra_member_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "stage"
            m.stage_panel(self.plan, root, self.catalogs, self.opener([]))
            complete = root / "COMPLETE"
            original = next(complete.rglob("*.flac")); raw = original.read_bytes()
            original.write_bytes(b"X" + raw[1:])
            with self.assertRaises(ValueError):
                m.read_release(complete, self.plan, self.catalogs)
            original.write_bytes(raw)
            (complete / "extra.flac").write_bytes(b"SEALED RESERVE FIXTURE")
            with self.assertRaises(ValueError):
                m.read_release(complete, self.plan, self.catalogs)

    def test_module_has_no_network_or_audio_or_execution_opener(self):
        tree = ast.parse(SCRIPT.read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(n.name.split(".")[0] for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module.split(".")[0])
        self.assertLessEqual(imports, {"__future__", "collections", "contextlib", "gzip", "hashlib", "json", "os", "pathlib", "re", "tarfile", "time"})
        self.assertFalse(any(isinstance(x, ast.Assert) for x in ast.walk(tree)))
        self.assertNotIn("__main__", SCRIPT.read_text())
        self.assertNotIn("extractall", SCRIPT.read_text())


if __name__ == "__main__":
    unittest.main()
