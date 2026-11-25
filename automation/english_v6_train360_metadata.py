"""Bounded, metadata-only acquisition of the original TRAIN-clean-360 archive.

The complete compressed stream and every opaque FLAC member are hashed. Only
UTF-8 metadata and member identities are retained. No audio decoder, model,
eligibility predicate, ranking, assessment, or scientific panel is imported.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import resource
import shutil
import signal
import subprocess
import tarfile
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REPO = "jiying2007/kws-data"
BRANCH = "refs/heads/research/en-v6-train360-metadata-once-20261125"
WORKFLOW = ".github/workflows/english-v6-train360-metadata-once.yml"
TEMPLATE = "automation/english_v6_train360_metadata_workflow_template.yml"
TOKEN = "__TRAIN360_METADATA_PREPARED_PARENT__"
MARKER = "english-v6-train360-metadata-consumed-20261125"
REGISTRATION = "automation/english_v6_train360_metadata_registration.json"
CENSUS = "automation/english_v6_b21_panels/SPEAKERS.TXT"
DOCS = "research/librispeech_train360_metadata_20261125/"
FROZEN = {
    REGISTRATION: "bf973f4be99cfa73610d1d7da543a19517da661840a28d78b99f7bc3e7a6222a",
    CENSUS: "3350f093382f8baa1d589c9f5a5cee8ec3705e656f521d6eafbaa31a5fcd8f76",
    DOCS + "OFFICIAL_HEAD.json": "7a103bdc958b3aa67f37887469e81918fa2c7a114a39568326ecfd50984a373e",
    DOCS + "OFFICIAL_SOURCES.json": "27a7930a87f87a9a3b2a3cdf2a6abab87a2939d60d46ee39e19c8b58688a9bc8",
    DOCS + "TORCHAUDIO_LIBRISPEECH.py.txt": "ce4cc91567215ace0092d60023e8219961f865e27ce02d2e0e8f44d7fa538122",
    DOCS + "OPENSLR_MD5SUM.txt": "efe6522682d078fdcfd2c3b2d418a53a89683b342dd5147a769a41a7e021fe3f",
    DOCS + "SOURCE_EVIDENCE.json": "f1e5ab5ef21bea8e47b2688d1160979eb17a062ddc58b4815f29e9cbfa9345b4",
}
OWN = (
    "automation/english_v6_train360_metadata.py",
    "automation/test_english_v6_train360_metadata.py",
    TEMPLATE,
    WORKFLOW,
    ".github/workflows/english-v6-train360-metadata-contracts.yml",
    DOCS + "README.md",
)
SPLIT = "train-clean-360"
PREFIX = "LibriSpeech/" + SPLIT + "/"
TRANSCRIPT = re.compile(PREFIX + r"([1-9][0-9]*)/([1-9][0-9]*)/([1-9][0-9]*)-([1-9][0-9]*)\.trans\.txt")
FLAC = re.compile(PREFIX + r"([1-9][0-9]*)/([1-9][0-9]*)/([1-9][0-9]*)-([1-9][0-9]*)-([0-9]+)\.flac")
DIRECTORY = re.compile(r"LibriSpeech(?:/train-clean-360(?:/[1-9][0-9]*(?:/[1-9][0-9]*)?)?)?")
ROOT_TEXT = frozenset("LibriSpeech/" + n for n in
    ("SPEAKERS.TXT", "README.TXT", "LICENSE.TXT", "BOOKS.TXT", "CHAPTERS.TXT"))


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def record(path, value, maximum=8388608):
    """Exclusive bounded write; a failed serialization remains visibly partial."""
    path = Path(path)
    need(not path.exists() and not path.is_symlink(), "fresh output: " + str(path))
    partial = path.with_name(path.name + ".INCOMPLETE")
    count = 0
    with partial.open("xb") as out:
        encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, indent=2)
        for part in encoder.iterencode(value):
            raw = part.encode()
            count += len(raw)
            need(count <= maximum, "serialized output byte limit: " + path.name)
            out.write(raw)
        count += 1
        need(count <= maximum, "serialized output byte limit: " + path.name)
        out.write(b"\n")
        out.flush()
        os.fsync(out.fileno())
    partial.rename(path)
    return dict(bytes=count, sha256=file_digest(path))


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def registration(root=ROOT):
    root = Path(root)
    for name, expected in FROZEN.items():
        path = root / name
        need(path.is_file() and not path.is_symlink() and file_digest(path) == expected,
             "frozen metadata-only input: " + name)
    value = json.loads((root / REGISTRATION).read_bytes())
    need(value["state"] == "PREPARED_DISABLED" and value["split"] == SPLIT
         and value["candidate_id"] is value["assessment"] is value["selected_people"] is None
         and value["source_opening_authorized"] is value["model_execution_authorized"] is False,
         "metadata acquisition gives no scientific or selection authorization")
    census_ids((root / CENSUS).read_bytes(), value)
    return value


def paths():
    return sorted(set(FROZEN) | set(OWN))


def census_ids(raw, spec):
    need(digest(raw) == spec["completeness"]["official_speakers_sha256"], "complete original SPEAKERS.TXT SHA")
    all_ids, selected_split = set(), set()
    for line in raw.decode("utf-8", errors="strict").splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        fields = [field.strip() for field in line.split("|")]
        need(len(fields) >= 3 and re.fullmatch(r"[1-9][0-9]*", fields[0])
             and fields[0] not in all_ids, "unique original speaker census rows")
        all_ids.add(fields[0])
        if fields[2] == SPLIT:
            selected_split.add(fields[0])
    need(len(selected_split) == spec["completeness"]["split_census_readers"], "full upstream TRAIN split census")
    return selected_split


class CompressedReader:
    """Hash each received byte once; neither seeking nor a disk cache exists."""
    def __init__(self, source, spec, progress=None, clock=time.monotonic, started=None):
        self.source, self.spec, self.progress, self.clock = source, spec, progress, clock
        self.started = clock() if started is None else started
        self.deadline = self.started + spec["transport"]["maximum_elapsed_seconds"]
        self.next_progress = self.started
        self.count, self.expanded, self.eof = 0, 0, False
        self.sha, self.md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)

    def check(self):
        now = self.clock()
        need(now <= self.deadline, "whole transfer/parse/hash deadline exceeded")
        if self.progress is not None and now >= self.next_progress:
            elapsed = now - self.started
            self.progress(dict(status="INCOMPLETE", elapsed_seconds=elapsed,
                compressed_bytes=self.count, expanded_bytes=self.expanded,
                compressed_bytes_per_second=self.count / elapsed if elapsed else 0))
            self.next_progress = now + self.spec["limits"]["progress_interval_seconds"]

    def read(self, size=-1):
        self.check()
        need(isinstance(size, int) and size >= 0, "bounded compressed read required")
        if not size:
            return b""
        size = min(size, self.spec["transport"]["read_chunk_bytes"],
                   self.spec["corpus"]["bytes"] - self.count + 1)
        raw = self.source.read(size)
        need(isinstance(raw, bytes) and len(raw) <= size, "bounded byte stream response")
        self.count += len(raw)
        need(self.count <= self.spec["corpus"]["bytes"], "extra compressed bytes")
        self.sha.update(raw)
        self.md5.update(raw)
        self.eof = not raw
        self.check()
        return raw


class ExpandedReader:
    def __init__(self, compressed):
        self.compressed = compressed
        self.gzip = gzip.GzipFile(fileobj=compressed, mode="rb")

    def read(self, size):
        self.compressed.check()
        need(0 <= size <= self.compressed.spec["transport"]["read_chunk_bytes"], "bounded expanded read")
        raw = self.gzip.read(size)
        self.compressed.expanded += len(raw)
        need(self.compressed.expanded <= self.compressed.spec["limits"]["expanded_tar_bytes"],
             "all expanded tar headers/payload/padding byte cap")
        self.compressed.check()
        return raw

    def exact(self, size):
        raw = self.read(size)
        need(len(raw) == size, "truncated tar header, payload, or padding")
        return raw


def member_kind(member, spec):
    name = member.name
    need(0 < len(name.encode()) <= spec["limits"]["member_path_bytes"]
         and "\\" not in name and "\x00" not in name and not name.startswith("/")
         and str(PurePosixPath(name)) == name and ".." not in PurePosixPath(name).parts,
         "canonical bounded relative member path")
    if member.isdir():
        need(DIRECTORY.fullmatch(name) is not None and member.size == 0, "TRAIN-only directory")
        return "directory"
    need(member.isfile() and member.type in (tarfile.REGTYPE, tarfile.AREGTYPE),
         "only ordinary short-path files and directories; no links/PAX/sparse/extensions")
    if name in ROOT_TEXT:
        kind = "root_text"
    elif TRANSCRIPT.fullmatch(name):
        speaker, chapter, sp2, ch2 = TRANSCRIPT.fullmatch(name).groups()
        need((speaker, chapter) == (sp2, ch2), "canonical transcript speaker/chapter")
        kind = "transcript"
    else:
        match = FLAC.fullmatch(name)
        need(match is not None and match.group(1, 2) == match.group(3, 4),
             "only original TRAIN-clean-360 transcript or FLAC paths")
        kind = "opaque_flac"
    cap = spec["limits"]["single_flac_bytes" if kind == "opaque_flac" else "single_text_bytes"]
    need(0 < member.size <= cap, "ordinary member byte cap")
    return kind


def validate_catalog(catalog, index, spec, census_raw):
    """Offline text/member identity checks; deliberately has no eligibility code."""
    census = census_ids(census_raw, spec)
    need(catalog["schema"] == "english-v6-train360-complete-text-catalog-v1"
         and index["schema"] == "english-v6-train360-complete-member-index-v1"
         and catalog["split"] == index["split"] == SPLIT
         and catalog["corpus"] == index["corpus"] == spec["corpus"]
         and catalog["metadata_only_counters"] == index["metadata_only_counters"] == spec["metadata_only_counters"]
         and catalog["selected_people"] is None and catalog["eligibility_computed"] is False,
         "exact metadata-only catalog identity")
    members = index["members"]
    need(0 < len(members) <= spec["limits"]["archive_members"]
         and [v["path"] for v in members] == sorted(set(v["path"] for v in members)),
         "bounded sorted unique complete member index")
    by_path, audio_ids, audio_readers, text_paths = {}, set(), set(), set()
    offset = 0
    for item in sorted(members, key=lambda x: x["tar_header_offset"]):
        need(type(item["bytes"]) is int and item["bytes"] >= 0 and item["tar_header_offset"] == offset,
             "complete contiguous original tar header/payload/padding coverage")
        offset += 512 + item["bytes"] + (-item["bytes"]) % 512
    need(offset + 1024 <= spec["limits"]["expanded_tar_bytes"], "room for both original tar end blocks")
    for item in members:
        need(set(item) == {"path", "kind", "bytes", "sha256", "tar_header_sha256", "tar_header_offset"}
             and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
             and re.fullmatch(r"[0-9a-f]{64}", item["tar_header_sha256"])
             and type(item["tar_header_offset"]) is int and item["tar_header_offset"] >= 0,
             "complete original member hash identity")
        member = tarfile.TarInfo(item["path"])
        member.size = item["bytes"]
        member.type = tarfile.DIRTYPE if item["kind"] == "directory" else tarfile.REGTYPE
        need(member_kind(member, spec) == item["kind"], "member kind/path/size matches")
        need(item["kind"] != "directory" or item["sha256"] == digest(b""), "empty directory payload hash")
        by_path[item["path"]] = item
        if item["kind"] == "opaque_flac":
            match = FLAC.fullmatch(item["path"])
            ident = "-".join(match.group(3, 4, 5))
            need(ident not in audio_ids, "unique FLAC source identity")
            audio_ids.add(ident)
            audio_readers.add(match.group(1))
        elif item["kind"] != "directory":
            text_paths.add(item["path"])
    texts = catalog["root_metadata"] + catalog["transcripts"]
    need([v["path"] for v in catalog["root_metadata"]] == sorted(ROOT_TEXT)
         and [v["path"] for v in catalog["transcripts"]] == sorted(set(v["path"] for v in catalog["transcripts"]))
         and set(v["path"] for v in texts) == text_paths, "every original text member retained exactly once")
    transcript_ids, transcript_readers, text_bytes = set(), set(), 0
    for item in texts:
        raw = item["text"].encode("utf-8", errors="strict")
        need(set(item) == {"path", "bytes", "sha256", "text"}
             and len(raw) == item["bytes"] == by_path[item["path"]]["bytes"]
             and digest(raw) == item["sha256"] == by_path[item["path"]]["sha256"],
             "exact original text bytes/hash and member binding")
        text_bytes += len(raw)
        if item["path"] == "LibriSpeech/SPEAKERS.TXT":
            need(raw == census_raw, "archive census equals the already authenticated original census")
        if by_path[item["path"]]["kind"] != "transcript":
            continue
        speaker, chapter, _, _ = TRANSCRIPT.fullmatch(item["path"]).groups()
        need(speaker in census, "only the original TRAIN split census")
        transcript_readers.add(speaker)
        lines = item["text"].splitlines()
        need(lines, "nonempty transcript file")
        for line in lines:
            need(0 < len(line.encode()) <= spec["limits"]["transcript_line_bytes"], "bounded transcript line")
            fields = line.split(" ", 1)
            need(len(fields) == 2 and fields[1] and re.fullmatch(speaker + "-" + chapter + r"-[0-9]+", fields[0])
                 and fields[0] not in transcript_ids, "unique original utterance identity with unmodified text")
            transcript_ids.add(fields[0])
    need(transcript_readers == audio_readers == census and transcript_ids == audio_ids,
         "all census readers and exact complete transcript/FLAC bijection")
    need(text_bytes == catalog["metadata_bytes"] <= spec["limits"]["all_text_bytes"], "complete retained text byte count")
    return dict(census_readers=len(census), transcript_members=len(catalog["transcripts"]),
        utterances=len(transcript_ids), flac_members=len(audio_ids), metadata_bytes=text_bytes,
        archive_members=len(members), opaque_flac_bytes=sum(x["bytes"] for x in members if x["kind"] == "opaque_flac"),
        tar_end_blocks_offset=offset, eligible_people=None, selected_people=None, eligibility_computed=False)


def scan_stream(source, spec, census_raw, progress=None, clock=time.monotonic, started=None, operational=None):
    """Parse only tar containers and UTF-8 text; opaque FLAC bytes are only hashed."""
    compressed = CompressedReader(source, spec, progress, clock, started)
    expanded = ExpandedReader(compressed)
    members, roots, transcripts, seen = [], [], [], set()
    text_bytes, opaque_bytes, complete = 0, 0, False
    try:
        while True:
            offset = compressed.expanded
            header = expanded.exact(512)
            if header == bytes(512):
                need(expanded.exact(512) == bytes(512), "two complete tar end blocks")
                while True:
                    tail = expanded.read(spec["transport"]["read_chunk_bytes"])
                    if not tail:
                        break
                    need(not tail.strip(b"\0"), "only tar zero padding after end blocks")
                break
            need(len(members) < spec["limits"]["archive_members"], "all tar headers count toward member cap")
            member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            kind = member_kind(member, spec)
            need(member.name not in seen, "duplicate archive member")
            seen.add(member.name)
            is_text = kind in ("root_text", "transcript")
            if is_text:
                text_bytes += member.size
                need(text_bytes <= spec["limits"]["all_text_bytes"], "all raw text byte cap")
            sha, chunks, remaining = hashlib.sha256(), [], member.size
            while remaining:
                part = expanded.exact(min(remaining, spec["transport"]["read_chunk_bytes"]))
                sha.update(part)
                if is_text:
                    chunks.append(part)
                elif kind == "opaque_flac":
                    opaque_bytes += len(part)
                remaining -= len(part)
            padding = expanded.exact((-member.size) % 512)
            need(not padding.strip(b"\0"), "canonical zero member padding")
            item = dict(path=member.name, kind=kind, bytes=member.size, sha256=sha.hexdigest(),
                        tar_header_sha256=digest(header), tar_header_offset=offset)
            members.append(item)
            if is_text:
                raw = b"".join(chunks)
                item = dict(path=member.name, bytes=len(raw), sha256=digest(raw), text=raw.decode("utf-8", errors="strict"))
                (roots if kind == "root_text" else transcripts).append(item)
        need(compressed.read(1) == b"" and compressed.eof, "complete HTTP/compressed EOF after gzip CRC/trailer")
        need(compressed.count == spec["corpus"]["bytes"]
             and compressed.sha.hexdigest() == spec["corpus"]["sha256"]
             and compressed.md5.hexdigest() == spec["corpus"]["md5"], "complete original corpus bytes/SHA256/official MD5")
        compressed.check()
        catalog = dict(schema="english-v6-train360-complete-text-catalog-v1", split=SPLIT, corpus=spec["corpus"],
            root_metadata=sorted(roots, key=lambda x: x["path"]), transcripts=sorted(transcripts, key=lambda x: x["path"]),
            metadata_bytes=text_bytes, metadata_only_counters=spec["metadata_only_counters"],
            eligibility_computed=False, selected_people=None)
        index = dict(schema="english-v6-train360-complete-member-index-v1", split=SPLIT, corpus=spec["corpus"],
            members=sorted(members, key=lambda x: x["path"]), metadata_only_counters=spec["metadata_only_counters"])
        summary = validate_catalog(catalog, index, spec, census_raw)
        need(summary["tar_end_blocks_offset"] == offset and compressed.expanded % 512 == 0,
             "complete block-aligned tar including its end and trailing padding")
        compressed.check()
        elapsed = clock() - compressed.started
        summary.update(compressed_bytes=compressed.count, compressed_sha256=compressed.sha.hexdigest(),
            compressed_md5=compressed.md5.hexdigest(), expanded_tar_bytes=compressed.expanded,
            gzip_crc_and_trailer_verified=True, compressed_eof_verified=True,
            elapsed_seconds=elapsed, compressed_bytes_per_second=compressed.count / elapsed if elapsed else 0)
        complete = True
        return catalog, index, summary
    finally:
        expanded.gzip.close()
        if operational is not None:
            elapsed = clock() - compressed.started
            operational.update(status="COMPLETE_AUTHENTICATED_METADATA" if complete else "INCOMPLETE",
                compressed_bytes=compressed.count, expanded_tar_bytes=compressed.expanded,
                compressed_sha256_observed=compressed.sha.hexdigest(), compressed_md5_observed=compressed.md5.hexdigest(),
                opaque_flac_payload_bytes_hashed=opaque_bytes, archive_headers_seen=len(members),
                elapsed_seconds=elapsed, compressed_bytes_per_second=compressed.count / elapsed if elapsed else 0)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        error = urllib.error.HTTPError(request.full_url, code,
            "fixed origin: redirects are not authorized", headers, fp)
        error.redirect_rejected = True
        raise error


def save_http_response(receipts, response):
    """Retain bounded response headers before validation; never read its body."""
    record(receipts / "HTTP_RESPONSE.json", dict(url=response.url, status=response.status,
        headers=list(response.headers.items()), body_saved=False, redirects=0, retries=0,
        redirect_rejected=bool(getattr(response, "redirect_rejected", False))))


def validate_response(response, spec):
    headers = response.headers
    need(response.status == 200 and response.url == spec["corpus"]["url"]
         and headers.get_all("Content-Length") == [str(spec["corpus"]["bytes"])]
         and headers.get("Content-Range") is None and headers.get("Transfer-Encoding") is None
         and headers.get("Content-Encoding", "identity").lower() == "identity",
         "one fixed full identity-encoded response with exact Content-Length")


def identity(prepared_parent, env=None, event=None, parent=None):
    env = os.environ if env is None else env
    need(all(env.get(k) == v for k, v in {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": REPO,
        "GITHUB_REF": BRANCH, "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ATTEMPT": "1"}.items()),
        "separately registered first public metadata push only")
    value = dict(run_id=env.get("GITHUB_RUN_ID", ""), run_head=env.get("GITHUB_SHA", ""),
                 run_branch=BRANCH, run_attempt=1, prepared_parent=prepared_parent)
    need(re.fullmatch(r"[1-9][0-9]*", value["run_id"])
         and all(re.fullmatch(r"[0-9a-f]{40}", value[k]) for k in ("run_head", "prepared_parent"))
         and value["run_head"] != value["prepared_parent"], "concrete separate metadata execution identity")
    if event is None:
        event = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_bytes())
    if parent is None:
        parent = subprocess.check_output(["git", "rev-parse", "HEAD^1"], cwd=env["GITHUB_WORKSPACE"], text=True).strip()
    need(event.get("before") == parent == prepared_parent and event.get("after") == value["run_head"]
         and event.get("ref") == BRANCH and event.get("forced") is event.get("deleted") is False
         and event.get("repository", {}).get("full_name") == REPO and event["repository"].get("private") is False,
         "exact reviewed parent and one ordinary public activation push")
    return value


def git_files(execution, root=ROOT):
    root = Path(root)
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)
    need(git("rev-parse", "HEAD").decode().strip() == execution["run_head"]
         and git("show", "-s", "--format=%P", "HEAD").decode().strip().split() == [execution["prepared_parent"]]
         and git("diff", "--name-only", execution["prepared_parent"], "HEAD").decode().splitlines() == [WORKFLOW],
         "single workflow-only activation commit from reviewed preparation")
    need((root / WORKFLOW).read_bytes() == (root / TEMPLATE).read_text().replace(TOKEN, execution["prepared_parent"]).encode(),
         "exact reviewed metadata-only activation template")
    files = []
    for name in paths():
        path = root / name
        raw = path.read_bytes()
        need(path.is_file() and not path.is_symlink() and raw == git("cat-file", "blob", "HEAD:" + name),
             "complete exact Git producer file: " + name)
        files.append(dict(path=name, bytes=len(raw), sha256=digest(raw),
            git_blob_sha1=hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()))
    return files


def gh(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", "repos/" + REPO + "/" + endpoint], timeout=90))


def once(execution, consumed, api=gh):
    branch = BRANCH.removeprefix("refs/heads/")
    runs = api("actions/workflows/" + WORKFLOW.rsplit("/", 1)[1] + "/runs?branch=" + branch + "&event=push&per_page=100")
    need(runs["total_count"] == len(runs["workflow_runs"]) == 1, "one metadata acquisition run, including failed runs")
    run = runs["workflow_runs"][0]
    need(str(run["id"]) == execution["run_id"] and run["head_sha"] == execution["run_head"]
         and run["head_branch"] == branch and run["run_attempt"] == 1 and run["event"] == "push"
         and run["repository"]["full_name"] == REPO and run["repository"]["private"] is False,
         "same separately registered public metadata-only run")
    artifacts = api("actions/artifacts?name=" + MARKER + "&per_page=100")
    need(artifacts["total_count"] == len(artifacts["artifacts"]) == int(consumed), "separate one-time metadata transport marker")
    if consumed:
        marker = artifacts["artifacts"][0]
        need(marker["name"] == MARKER and marker["expired"] is False
             and str(marker["workflow_run"]["id"]) == execution["run_id"]
             and marker["workflow_run"]["head_sha"] == execution["run_head"], "current metadata run owns marker")
    return dict(runs=runs, artifacts=artifacts)


def prepare(directory, prepared_parent):
    spec = registration()
    execution = identity(prepared_parent)
    files = git_files(execution)
    live = once(execution, False)
    directory = Path(directory)
    need(directory.is_absolute() and directory.parent.is_dir() and not directory.exists(), "fresh exclusive metadata destination")
    need(sum(item["bytes"] for item in files) + 1048576 < spec["limits"]["producer_bytes"], "producer capacity before transfer")
    directory.mkdir()
    for name in ("producer", "receipts", "consumed"):
        (directory / name).mkdir()
    for item in files:
        target = directory / "producer/repository" / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / item["path"], target)
    proof = dict(schema="english-v6-train360-metadata-producer-v1", execution=execution, files=files,
        frozen_inputs=FROZEN, metadata_only_counters=spec["metadata_only_counters"])
    record(directory / "producer/PRODUCER.json", proof)
    authorization = dict(schema="english-v6-train360-metadata-authorization-v1", execution=execution,
        producer_sha256=file_digest(directory / "producer/PRODUCER.json"), registration_sha256=FROZEN[REGISTRATION],
        maximum_archive_gets=1, scientific_panel_budget_consumed=False,
        metadata_only_counters=spec["metadata_only_counters"])
    record(directory / "receipts/AUTHORIZATION.json", authorization)
    record(directory / "receipts/PREPARATION_API.json", live)
    record(directory / "consumed/CONSUMPTION.json", dict(authorization, status="METADATA_TRANSPORT_CONSUMED_UNKNOWN"))
    return authorization


def validate_capture(directory, execution, spec):
    directory = Path(directory)
    need(ROOT.resolve() == (directory / "producer/repository").resolve(), "collection runs only from captured producer")
    proof = json.loads((directory / "producer/PRODUCER.json").read_bytes())
    authorization = json.loads((directory / "receipts/AUTHORIZATION.json").read_bytes())
    consumed = json.loads((directory / "consumed/CONSUMPTION.json").read_bytes())
    need(proof["schema"] == "english-v6-train360-metadata-producer-v1" and proof["execution"] == execution
         and proof["frozen_inputs"] == FROZEN and proof["metadata_only_counters"] == spec["metadata_only_counters"]
         and [x["path"] for x in proof["files"]] == paths(), "closed exact captured metadata producer")
    actual = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() or p.is_symlink())
    need(actual == paths() and not any(p.is_symlink() for p in ROOT.rglob("*")), "no missing/extra/symlink captured producer resource")
    for item in proof["files"]:
        path = ROOT / item["path"]
        raw = path.read_bytes()
        need(not path.is_symlink() and len(raw) == item["bytes"] and digest(raw) == item["sha256"]
             and hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == item["git_blob_sha1"],
             "unchanged captured producer byte identity")
    expected = dict(schema="english-v6-train360-metadata-authorization-v1", execution=execution,
        producer_sha256=file_digest(directory / "producer/PRODUCER.json"), registration_sha256=FROZEN[REGISTRATION],
        maximum_archive_gets=1, scientific_panel_budget_consumed=False,
        metadata_only_counters=spec["metadata_only_counters"])
    need(authorization == expected and consumed == dict(expected, status="METADATA_TRANSPORT_CONSUMED_UNKNOWN"),
         "exact metadata-only authorization and separate transport marker")
    return authorization


def collect(directory, prepared_parent):
    spec = registration()
    execution = identity(prepared_parent)
    directory = Path(directory)
    need(directory.is_absolute() and directory.is_dir() and not directory.is_symlink(), "existing exclusive metadata destination")
    validate_capture(directory, execution, spec)
    live = once(execution, True)
    receipts = directory / "receipts"
    record(receipts / "COLLECTION_API.json", live)
    limits = spec["limits"]
    available = os.statvfs(directory).f_bavail * os.statvfs(directory).f_frsize
    need(available >= limits["minimum_free_disk_bytes"], "bounded metadata-only disk capacity")
    resource.setrlimit(resource.RLIMIT_AS, (limits["address_space_bytes"], limits["address_space_bytes"]))
    record(receipts / "TRANSFER_STARTED.json", dict(status="INCOMPLETE", execution=execution,
        corpus=spec["corpus"], transport=spec["transport"], limits=limits,
        scientific_panel_budget_consumed=False, metadata_only_counters=spec["metadata_only_counters"]))
    result = dict(schema="english-v6-train360-metadata-acquisition-result-v1", execution=execution,
        status="METADATA_ACQUISITION_INCOMPLETE", scientific_panel_budget_consumed=False,
        algorithm_result=None, metadata_only_counters=spec["metadata_only_counters"],
        archive_get_attempted=0, archive_response_returned=0)
    started = time.monotonic()
    operational = dict(status="INCOMPLETE", compressed_bytes=0, expanded_tar_bytes=0,
                       opaque_flac_payload_bytes_hashed=0)
    def expired(signum, frame):
        raise TimeoutError("metadata acquisition fixed 180-minute wall deadline")
    previous_alarm = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, spec["transport"]["maximum_elapsed_seconds"])
    try:
        with (receipts / "PROGRESS.jsonl").open("x", encoding="utf-8") as log:
            progress_count = 0
            def progress(value):
                nonlocal progress_count
                progress_count += 1
                need(progress_count <= 185, "bounded operational progress records")
                line = json.dumps(value, sort_keys=True) + "\n"
                need(len(line.encode()) <= 1024, "bounded operational progress line")
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            request = urllib.request.Request(spec["corpus"]["url"], headers={"Accept-Encoding": "identity"}, method="GET")
            opener = urllib.request.build_opener(NoRedirect())
            record(receipts / "GET_ATTEMPT.json", dict(
                schema="english-v6-train360-metadata-get-attempt-v1", execution=execution,
                attempt=1, url=request.full_url, method=request.get_method(),
                headers=request.header_items(), maximum_archive_gets=1, retries=0, redirects=0))
            result["archive_get_attempted"] = 1
            try:
                response = opener.open(request, timeout=spec["transport"]["socket_timeout_seconds"])
            except urllib.error.HTTPError as exc:
                result["archive_response_returned"] = 1
                try:
                    save_http_response(receipts, exc)
                finally:
                    exc.close()
                raise
            result["archive_response_returned"] = 1
            with response:
                save_http_response(receipts, response)
                validate_response(response, spec)
                catalog, index, summary = scan_stream(response, spec, (ROOT / CENSUS).read_bytes(), progress,
                                                     started=started, operational=operational)
        need(time.monotonic() - started <= spec["transport"]["maximum_elapsed_seconds"], "complete metadata within transfer/parse deadline")
        incomplete = receipts / "INCOMPLETE"
        incomplete.mkdir()
        catalog_pin = record(incomplete / "FULL_METADATA_CATALOG.json", catalog, limits["catalog_bytes"])
        index_pin = record(incomplete / "FULL_MEMBER_INDEX.json", index, limits["member_index_bytes"])
        validation = dict(schema="english-v6-train360-metadata-validation-v1", status="COMPLETE_AUTHENTICATED_METADATA",
            execution=execution, corpus=spec["corpus"], summary=summary, catalog=catalog_pin, member_index=index_pin,
            metadata_only_counters=spec["metadata_only_counters"], eligibility_computed=False, selected_people=None,
            source_policy_change=False, scientific_panel_budget_consumed=False, source_continuation_authorized=False)
        record(incomplete / "VALIDATION.json", validation)
        need(sum(p.stat().st_size for p in receipts.rglob("*") if p.is_file()) + 1048576 <= limits["receipts_bytes"],
             "all receipts fit the predeclared artifact role")
        need(sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) + 2097152 <= limits["total_artifact_bytes"],
             "total three artifact roles including ZIP/metadata reserve")
        incomplete.rename(receipts / "COMPLETE")
        result.update(status="COMPLETE_AUTHENTICATED_METADATA", validation_sha256=file_digest(receipts / "COMPLETE/VALIDATION.json"),
                      summary=summary, catalog=catalog_pin, member_index=index_pin)
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc)[:2048])
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        result["wall_elapsed_seconds"] = time.monotonic() - started
        result["transport_observation"] = operational
        record(receipts / "RESULT.json", result)
    return result


def verify_catalog(directory):
    spec = registration()
    directory = Path(directory)
    need(directory.name == "COMPLETE" and directory.is_dir() and not directory.is_symlink(), "authenticated complete metadata directory")
    need(sorted(p.name for p in directory.iterdir()) == ["FULL_MEMBER_INDEX.json", "FULL_METADATA_CATALOG.json", "VALIDATION.json"],
         "exact complete metadata artifact members")
    validation_path = directory / "VALIDATION.json"
    need(validation_path.is_file() and not validation_path.is_symlink() and validation_path.stat().st_size <= 1048576,
         "bounded ordinary validation receipt")
    validation = json.loads(validation_path.read_bytes())
    for name, key, cap in (("FULL_METADATA_CATALOG.json", "catalog", "catalog_bytes"),
                           ("FULL_MEMBER_INDEX.json", "member_index", "member_index_bytes")):
        path = directory / name
        need(path.is_file() and not path.is_symlink() and path.stat().st_size <= spec["limits"][cap]
             and dict(bytes=path.stat().st_size, sha256=file_digest(path)) == validation[key], "exact bounded retained metadata bytes")
    catalog = json.loads((directory / "FULL_METADATA_CATALOG.json").read_bytes())
    index = json.loads((directory / "FULL_MEMBER_INDEX.json").read_bytes())
    summary = validate_catalog(catalog, index, spec, (ROOT / CENSUS).read_bytes())
    need(validation["schema"] == "english-v6-train360-metadata-validation-v1"
         and validation["status"] == "COMPLETE_AUTHENTICATED_METADATA" and validation["corpus"] == spec["corpus"]
         and all(validation["summary"][k] == v for k, v in summary.items())
         and validation["summary"]["compressed_bytes"] == spec["corpus"]["bytes"]
         and validation["summary"]["compressed_sha256"] == spec["corpus"]["sha256"]
         and validation["summary"]["compressed_md5"] == spec["corpus"]["md5"]
         and summary["tar_end_blocks_offset"] + 1024 <= validation["summary"]["expanded_tar_bytes"] <= spec["limits"]["expanded_tar_bytes"]
         and validation["summary"]["gzip_crc_and_trailer_verified"] is validation["summary"]["compressed_eof_verified"] is True
         and validation["eligibility_computed"] is False and validation["selected_people"] is None,
         "complete original-corpus authenticated catalog report")
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    for command in ("prepare", "collect"):
        part = sub.add_parser(command)
        part.add_argument("directory")
        part.add_argument("prepared_parent")
    check = sub.add_parser("verify-catalog")
    check.add_argument("directory")
    args = parser.parse_args()
    if args.command == "check":
        spec = registration()
        value = dict(status="PREPARED_DISABLED", captured_paths=paths(), split=spec["split"], source_or_model_calls=0)
    elif args.command == "verify-catalog":
        value = verify_catalog(args.directory)
    else:
        value = globals()[args.command](args.directory, args.prepared_parent)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
