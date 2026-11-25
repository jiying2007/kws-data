"""Draft B28 whole-original staging for exactly the current frozen natural panel.

No CLI, lineage override, decoder, model or source-budget opener is supplied.
The future B28 caller must authenticate its original training/assessment proof
and consume its separately registered staging marker BEFORE calling stage_panel.
Metadata is fixed before training; only that panel's 24 opaque FLAC originals
are staged. Both complete corpus streams must authenticate before any release.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import time


CORPORA = {
    "train-clean-100": {
        "split": "train-clean-100",
        "url": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "bytes": 6387309499,
        "sha256": "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",
        "md5": "2a93770f6d5c6c964bc36631d331a522",
    },
    "train-clean-360": {
        "split": "train-clean-360",
        "url": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "bytes": 23049477885,
        "sha256": "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf",
        "md5": "c0e676e450a7ff2f54aeade5171606fa",
    },
}
SPEAKERS_SHA = "3350f093382f8baa1d589c9f5a5cee8ec3705e656f521d6eafbaa31a5fcd8f76"
BLOCK = frozenset("HEY HAY MERO MEERO MIRO MARROW MERROW".split())
ROOT_TEXT = frozenset("LibriSpeech/" + n for n in
    ("SPEAKERS.TXT", "README.TXT", "LICENSE.TXT", "BOOKS.TXT", "CHAPTERS.TXT"))
LIMITS = {
    "chunk_bytes": 1048576,
    "expanded_bytes_per_corpus": 42949672960,
    "member_headers_per_corpus": 150000,
    "single_original_bytes": 134217728,
    "selected_original_bytes_total": 134217728,
    "single_text_bytes": 8388608,
    "all_text_bytes_per_corpus": 67108864,
    "path_bytes": 256,
    "maximum_streams": 2,
    "maximum_panel_seconds": 14400,
    "minimum_free_disk_bytes": 268435456,
}
ROW_FIELDS = {"archive_key", "corpus_split", "speaker", "chapter", "source_id", "transcript",
              "transcript_file_sha256", "transcript_member", "flac_member", "original_flac"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def write(path, value):
    raw = encoded(value)
    need(len(raw) <= 8388608, "bounded original receipt")
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def is_hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def canonical_person(value):
    need(type(value) is str and re.fullmatch(r"[1-9][0-9]*", value), "canonical original person")
    return value


def text_eligible(text):
    words = text.split(" ")
    return 4 <= len(words) <= 14 and not BLOCK.intersection(words) and all(
        re.fullmatch(r"[A-Z]+(?:'[A-Z]+)*", word) for word in words)


def source_group(row):
    return "librispeech-" + row["corpus_split"] + "-" + row["speaker"]


def natural_id(panel, row):
    need(panel in ("primary", "replication"), "explicit current panel")
    return "b28-" + panel + "-natural-" + row["source_id"]


def census_by_split(raw):
    need(sha(raw) == SPEAKERS_SHA, "same complete original speaker census")
    groups, seen = defaultdict(set), set()
    for line in raw.decode("utf-8", "strict").splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        fields = [x.strip() for x in line.split("|")]
        need(len(fields) >= 3 and fields[0] not in seen, "unique original census row")
        seen.add(canonical_person(fields[0]))
        groups[fields[2]].add(fields[0])
    return groups


def metadata_rows(catalog, split, census_raw, member_index=None):
    """Normalize already byte-pinned original catalogs, never HF conversions.

    The caller still binds complete original metadata-acquisition run/artifacts.
    This structural reader does not authenticate acquisition or grant execution.
    """
    need(split in CORPORA, "only the two registered original TRAIN corpora")
    corpus = CORPORA[split]
    census = census_by_split(census_raw)[split]
    if split == "train-clean-100":
        need(catalog["schema"] == "english-v6-b21-full-train-clean-100-text-metadata-v1"
             and catalog["corpus_sha256"] == corpus["sha256"] and catalog["corpus_bytes"] == corpus["bytes"]
             and catalog["speakers_text"].encode("utf-8") == census_raw
             and catalog["speakers_sha256"] == SPEAKERS_SHA and catalog["metadata_only"] is True,
             "unchanged original whole-clean100 catalog identity")
        need(member_index is None, "clean100 has no invented prior FLAC member-index proof")
        flac_pins = {}
    else:
        need(catalog["schema"] == "english-v6-train360-complete-text-catalog-v1"
             and catalog["split"] == split and catalog["corpus"] == corpus_without_split(corpus)
             and catalog["eligibility_computed"] is False and catalog["selected_people"] is None,
             "actual complete original clean360 catalog, no converted-text substitution")
        roots = {x["path"]: x for x in catalog["root_metadata"]}
        need(set(roots) == ROOT_TEXT and roots["LibriSpeech/SPEAKERS.TXT"]["text"].encode("utf-8") == census_raw,
             "complete original clean360 root metadata and same census")
        need(type(member_index) is dict and member_index["schema"] == "english-v6-train360-complete-member-index-v1"
             and member_index["split"] == split and member_index["corpus"] == catalog["corpus"],
             "actual original clean360 member index is required")
        all_members = member_index["members"]
        need(len(all_members) == len({x["path"] for x in all_members}), "unique complete clean360 members")
        flac_pins = {x["path"]: {"bytes": x["bytes"], "sha256": x["sha256"]}
                     for x in all_members if x["kind"] == "opaque_flac"}
    transcripts = catalog["transcripts"]
    need([x["path"] for x in transcripts] == sorted(set(x["path"] for x in transcripts)), "unique original transcript paths")
    rows, people, source_ids, flac_members = [], set(), set(), set()
    for item in transcripts:
        path = item["path"]
        match = re.fullmatch(r"LibriSpeech/" + split + r"/([1-9][0-9]*)/([1-9][0-9]*)/\1-\2\.trans\.txt", path)
        need(match is not None and match.group(1) in census, "correct split/speaker/chapter original transcript")
        speaker, chapter = match.groups()
        raw = item["text"].encode("utf-8", "strict")
        need(len(raw) == item["bytes"] and sha(raw) == item["sha256"], "exact complete original transcript bytes")
        people.add(speaker)
        for line in item["text"].splitlines():
            fields = line.split(" ", 1)
            need(len(fields) == 2 and fields[1] and re.fullmatch(speaker + "-" + chapter + r"-[0-9]+", fields[0])
                 and fields[0] not in source_ids, "unique original utterance identifier and text")
            ident, text = fields
            source_ids.add(ident)
            flac = "LibriSpeech/" + split + "/" + speaker + "/" + chapter + "/" + ident + ".flac"
            flac_members.add(flac)
            pin = flac_pins.get(flac)
            need(split == "train-clean-100" or type(pin) is dict, "each360 original has prior full-stream member identity")
            if pin is not None:
                need(type(pin["bytes"]) is int and 0 < pin["bytes"] <= LIMITS["single_original_bytes"]
                     and is_hash(pin["sha256"]), "bounded exact original FLAC pin")
            rows.append(dict(archive_key=split, corpus_split=split, speaker=speaker, chapter=chapter,
                source_id=ident, transcript=text, transcript_file_sha256=item["sha256"], transcript_member=path,
                flac_member=flac, original_flac=pin))
    need(people == census, "all corpus census readers represented before ranking")
    need(split == "train-clean-100" or flac_members == set(flac_pins), "complete360 transcript/FLAC bijection")
    return rows


def corpus_without_split(corpus):
    return {**{k: corpus[k] for k in ("url", "bytes", "sha256", "md5")}, "license": "CC BY 4.0"}


def normalized_catalog(catalog, split, census_raw, member_index=None):
    """Preserve every transcript; carry only text-eligible rows into ranking."""
    rows = metadata_rows(catalog, split, census_raw, member_index)
    return dict(schema="english-v6-b28-normalized-original-text-catalog-v1", split=split,
        corpus=CORPORA[split], speakers_text=census_raw.decode("utf-8"),
        transcripts=catalog["transcripts"], rows=[r for r in rows if text_eligible(r["transcript"])],
        original_catalog_canonical_sha256=sha(encoded(catalog)),
        original_member_index_canonical_sha256=sha(encoded(member_index)) if member_index is not None else None,
        original_metadata_acquisition_authentication_required_separately=True)


def select_natural_rows(all_rows, excluded_people, salt):
    """Same hash rank and text rule; no actual selection until both catalogs bind."""
    need(type(salt) is str and salt, "prospectively frozen nonempty salt")
    blocked = {canonical_person(p) for p in excluded_people}
    by_person, splits, ids = defaultdict(list), {}, set()
    for row in all_rows:
        need(set(row) == ROW_FIELDS and row["archive_key"] == row["corpus_split"] in CORPORA, "explicit corpus on every row")
        person = canonical_person(row["speaker"])
        need(person not in splits or splits[person] == row["corpus_split"], "same person cannot be fresh through split relabeling")
        splits[person] = row["corpus_split"]
        need(row["source_id"] not in ids, "globally unique natural original ID")
        ids.add(row["source_id"])
        if person not in blocked and text_eligible(row["transcript"]):
            by_person[person].append(row)
    eligible = {p for p, rows in by_person.items() if len(rows) >= 2}
    rank = lambda kind, value: (sha((salt + "/" + kind + "/" + value).encode()), value)
    speakers = sorted(eligible, key=lambda p: rank("speaker", p))[:12]
    need(len(speakers) == 12, "twelve independent eligible readers, no replacement")
    roster = [row for p in speakers for row in sorted(by_person[p], key=lambda x: rank("utterance", x["source_id"]))[:2]]
    return speakers, roster


def validate_plan(plan, catalogs):
    """A frozen plan is data, not an execution permit; gates belong to caller."""
    fields = {"schema", "panel", "assessment_sha256", "training_readback_sha256", "roster", "corpora", "catalog_pins", "limits"}
    need(set(plan) == fields and plan["schema"] == "english-v6-b28-multicorpus-staging-plan-v1"
         and plan["panel"] in ("primary", "replication") and is_hash(plan["assessment_sha256"])
         and is_hash(plan["training_readback_sha256"]) and plan["limits"] == LIMITS,
         "explicit frozen B28 current-panel staging identity and bounds")
    roster = plan["roster"]
    need(len(roster) == len({r["source_id"] for r in roster}) == 24
         and len({r["speaker"] for r in roster}) == 12
         and set(Counter(r["speaker"] for r in roster).values()) == {2}, "one global24-slot/12-reader roster")
    keys = sorted({r["archive_key"] for r in roster})
    need(0 < len(keys) <= 2 and sorted(plan["corpora"]) == sorted(catalogs) == sorted(plan["catalog_pins"]) == keys,
         "only actually selected corpora, no sealed-replica prefetch")
    for key in keys:
        need(key in CORPORA and plan["corpora"][key] == CORPORA[key], "exact registered original corpus URL/bytes/hash")
        catalog = catalogs[key]
        need(catalog["schema"] == "english-v6-b28-normalized-original-text-catalog-v1" and catalog["split"] == key
             and catalog["corpus"] == CORPORA[key] and sha(catalog["speakers_text"].encode()) == SPEAKERS_SHA
             and is_hash(catalog["original_catalog_canonical_sha256"])
             and catalog["original_metadata_acquisition_authentication_required_separately"] is True
             and (key != "train-clean-360" or is_hash(catalog["original_member_index_canonical_sha256"])),
             "whole original catalog identities, explicit independent acquisition requirement")
        need(plan["catalog_pins"][key] == {"bytes": len(encoded(catalogs[key])), "sha256": sha(encoded(catalogs[key]))},
             "complete normalized pretraining catalog bound to current SOURCE")
    people = {}
    for row in roster:
        need(set(row) == ROW_FIELDS and row["archive_key"] == row["corpus_split"] in CORPORA
             and text_eligible(row["transcript"]), "each fixed row uses unchanged native text eligibility")
        person = canonical_person(row["speaker"])
        chapter = canonical_person(row["chapter"])
        need(re.fullmatch(person + "-" + chapter + r"-[0-9]+", row["source_id"])
             and row["transcript_member"] == "LibriSpeech/" + row["corpus_split"] + "/" + person + "/" + chapter + "/" + person + "-" + chapter + ".trans.txt"
             and row["flac_member"] == "LibriSpeech/" + row["corpus_split"] + "/" + person + "/" + chapter + "/" + row["source_id"] + ".flac"
             and is_hash(row["transcript_file_sha256"]), "exact row-to-corpus-to-original-member mapping")
        pin = row["original_flac"]
        need((pin is None and row["corpus_split"] == "train-clean-100") or
             (type(pin) is dict and set(pin) == {"bytes", "sha256"} and type(pin["bytes"]) is int
              and 0 < pin["bytes"] <= LIMITS["single_original_bytes"] and is_hash(pin["sha256"])),
             "required preauthenticated360 whole-member identity")
        need(person not in people or people[person] == row["corpus_split"], "globally canonical people across splits")
        people[person] = row["corpus_split"]
        need(row in catalogs[row["archive_key"]]["rows"], "exact immutable original catalog row, no substitutions")
    return keys


class CountedStream:
    def __init__(self, stream, corpus, deadline, clock):
        self.stream, self.corpus, self.deadline, self.clock = stream, corpus, deadline, clock
        self.count, self.expanded = 0, 0
        self.sha256, self.md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)

    def read(self, size=-1):
        need(type(size) is int and size >= 0, "bounded compressed read")
        need(self.clock() <= self.deadline, "fixed complete-panel deadline")
        if not size:
            return b""
        size = min(size, LIMITS["chunk_bytes"], self.corpus["bytes"] - self.count + 1)
        raw = self.stream.read(size)
        need(type(raw) is bytes and len(raw) <= size, "bounded exact response bytes")
        self.count += len(raw)
        need(self.count <= self.corpus["bytes"], "extra compressed bytes")
        self.sha256.update(raw)
        self.md5.update(raw)
        need(self.clock() <= self.deadline, "complete-panel deadline after blocking read")
        return raw


def member_kind(member, split):
    name = member.name
    need(0 < len(name.encode()) <= LIMITS["path_bytes"] and "\\" not in name
         and str(PurePosixPath(name)) == name and not name.startswith("/") and ".." not in PurePosixPath(name).parts,
         "canonical bounded relative tar path")
    if member.isdir():
        need(re.fullmatch(r"LibriSpeech(?:/" + split + r"(?:/[1-9][0-9]*(?:/[1-9][0-9]*)?)?)?", name)
             and member.size == 0, "only current split directories")
        return "directory"
    need(member.isfile() and member.type in (tarfile.REGTYPE, tarfile.AREGTYPE), "ordinary files only, no links/PAX/extensions")
    if name in ROOT_TEXT:
        kind = "root_text"
    elif re.fullmatch(r"LibriSpeech/" + split + r"/([1-9][0-9]*)/([1-9][0-9]*)/\1-\2\.trans\.txt", name):
        kind = "transcript"
    else:
        need(re.fullmatch(r"LibriSpeech/" + split + r"/([1-9][0-9]*)/([1-9][0-9]*)/\1-\2-[0-9]+\.flac", name),
             "current split ordinary original FLAC path")
        kind = "opaque_flac"
    cap = LIMITS["single_original_bytes"] if kind == "opaque_flac" else LIMITS["single_text_bytes"]
    need(type(member.size) is int and 0 < member.size <= cap, "prospective member byte bound")
    return kind


def stage_archive(stream, corpus, rows, catalog, destination, deadline, clock=time.monotonic, progress=None,
                  remaining_bytes=None):
    """One whole compressed stream; partial opaque FLAC bytes confer no release."""
    destination = Path(destination)
    need(destination.is_dir() and not any(destination.iterdir()), "fresh exclusive corpus staging directory")
    fixed = {r["flac_member"]: r for r in rows}
    need(len(fixed) == len(rows) and rows and len(rows) <= 24, "distinct current-panel whole members")
    expected_text = {x["path"]: x for x in catalog["transcripts"]}
    need(len(expected_text) == len(catalog["transcripts"]), "unique full original text catalog")
    compressed = CountedStream(stream, corpus, deadline, clock)
    expanded = gzip.GzipFile(fileobj=compressed, mode="rb")
    seen, seen_text, staged, roots = set(), set(), {}, set()
    text_bytes, staged_bytes, opaque_bytes, written_bytes = 0, 0, 0, 0
    remaining_bytes = LIMITS["selected_original_bytes_total"] if remaining_bytes is None else remaining_bytes
    need(type(remaining_bytes) is int and 0 <= remaining_bytes <= LIMITS["selected_original_bytes_total"], "remaining global staging byte bound")
    def read(size, exact=False):
        need(0 <= size <= LIMITS["chunk_bytes"] and clock() <= deadline, "bounded expanded read and deadline")
        raw = expanded.read(size)
        compressed.expanded += len(raw)
        need(compressed.expanded <= LIMITS["expanded_bytes_per_corpus"], "expanded whole archive cap")
        need(not exact or len(raw) == size, "truncated tar payload/header/padding")
        return raw
    try:
        while True:
            offset = compressed.expanded
            header = read(512, True)
            if header == bytes(512):
                need(read(512, True) == bytes(512), "both complete tar end blocks")
                while True:
                    tail = read(LIMITS["chunk_bytes"])
                    if not tail:
                        break
                    need(not tail.strip(b"\0"), "only zero tar padding after end blocks")
                break
            need(len(seen) < LIMITS["member_headers_per_corpus"], "bounded complete member inventory")
            item = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            kind = member_kind(item, corpus["split"])
            need(item.name not in seen, "no duplicate archive member")
            seen.add(item.name)
            selected = item.name in fixed
            if selected:
                need(kind == "opaque_flac", "only frozen current-panel FLAC can be staged")
                staged_bytes += item.size
                need(staged_bytes <= remaining_bytes, "bounded selected original staging across both corpora")
            if kind in ("transcript", "root_text"):
                text_bytes += item.size
                need(text_bytes <= LIMITS["all_text_bytes_per_corpus"], "bounded full original metadata")
            digest, text, remaining = hashlib.sha256(), [], item.size
            out = (destination / (fixed[item.name]["source_id"] + ".flac")).open("xb") if selected else None
            try:
                while remaining:
                    part = read(min(remaining, LIMITS["chunk_bytes"]), True)
                    remaining -= len(part)
                    if kind == "opaque_flac":
                        opaque_bytes += len(part)
                    if kind in ("transcript", "root_text"):
                        text.append(part)
                    if selected or kind in ("transcript", "root_text"):
                        digest.update(part)
                    if out is not None:
                        out.write(part)
                        written_bytes += len(part)
                if out is not None:
                    out.flush()
                    os.fsync(out.fileno())
            finally:
                if out is not None:
                    out.close()
            need(not read((-item.size) % 512, True).strip(b"\0"), "canonical zero member padding")
            if selected:
                row = fixed[item.name]
                pin = {"bytes": item.size, "sha256": digest.hexdigest()}
                need(row["original_flac"] is None or row["original_flac"] == pin, "exact previously authenticated original member bytes")
                staged[item.name] = {**pin, "source_id": row["source_id"], "file": row["source_id"] + ".flac",
                    "member": item.name, "tar_header_sha256": sha(header), "tar_header_offset": offset}
            elif kind == "transcript":
                need(item.name in expected_text, "no unknown transcript outside full original catalog")
                actual = b"".join(text)
                original = expected_text[item.name]
                need(actual == original["text"].encode("utf-8") and len(actual) == original["bytes"]
                     and digest.hexdigest() == original["sha256"], "all original transcript bytes unchanged")
                seen_text.add(item.name)
            elif kind == "root_text":
                roots.add(item.name)
                if item.name == "LibriSpeech/SPEAKERS.TXT":
                    need(b"".join(text) == catalog["speakers_text"].encode("utf-8") and digest.hexdigest() == SPEAKERS_SHA,
                         "same complete census in actual corpus stream")
            if progress is not None:
                progress.update(compressed_bytes=compressed.count, expanded_bytes=compressed.expanded,
                    opaque_flac_payload_bytes_read=opaque_bytes, opaque_selected_originals_staged=len(staged),
                    selected_bytes_staged=staged_bytes)
        need(compressed.read(1) == b"" and compressed.count == corpus["bytes"]
             and compressed.sha256.hexdigest() == corpus["sha256"] and compressed.md5.hexdigest() == corpus["md5"],
             "complete compressed EOF/bytes/SHA256/MD5 after actual gzip CRC and trailer")
        need(set(staged) == set(fixed) and seen_text == set(expected_text) and roots == ROOT_TEXT,
             "all and only current-panel original members plus complete original metadata")
        need(compressed.expanded % 512 == 0 and clock() <= deadline, "complete block-aligned stream inside fixed time bound")
        return dict(schema="english-v6-b28-original-corpus-stream-v1", status="COMPLETE_VERIFIED",
            corpus=corpus, compressed_bytes=compressed.count, compressed_sha256=compressed.sha256.hexdigest(),
            compressed_md5=compressed.md5.hexdigest(), expanded_bytes=compressed.expanded,
            gzip_crc_and_trailer_verified=True, compressed_eof_verified=True, all_original_transcripts_verified=True,
            opaque_flac_payload_bytes_read=opaque_bytes, opaque_selected_originals_staged=len(staged),
            selected_bytes_staged=staged_bytes, selected_members=[staged[x] for x in sorted(staged)],
            archive_headers=len(seen), audio_headers_interpreted=0, native_decodes=0, tts_calls=0,
            observer_calls=0, model_calls=0, retries=0, original_compressed_archive_retained=False)
    finally:
        if progress is not None:
            progress.update(compressed_bytes=compressed.count, expanded_bytes=compressed.expanded,
                opaque_flac_payload_bytes_read=opaque_bytes, opaque_selected_originals_staged=len(staged),
                selected_bytes_written=written_bytes)
        expanded.close()


def validate_response(response, corpus):
    headers = response.headers
    need(response.status == 200 and response.url == corpus["url"]
         and headers.get_all("Content-Length") == [str(corpus["bytes"])]
         and headers.get("Content-Range") is None and headers.get("Transfer-Encoding") is None
         and headers.get("Content-Encoding", "identity").lower() == "identity",
         "one exact full original response, no ranges/redirect/chunk/encoding substitution")


def stage_panel(plan, directory, catalogs, open_response, clock=time.monotonic):
    """Low-level helper only; mandatory outer B28 lineage/marker gates are separate.

    open_response(corpus) must make exactly one bounded HTTPS GET, reject every
    redirect and enforce socket timeout. This helper calls it once per selected
    corpus with no retry; it has no built-in network adapter or executable CLI.
    """
    keys = validate_plan(plan, catalogs)
    directory = Path(directory)
    need(directory.is_absolute() and directory.parent.is_dir() and not directory.exists(), "fresh staging, never resume or retry")
    need(os.statvfs(directory.parent).f_bavail * os.statvfs(directory.parent).f_frsize >= LIMITS["minimum_free_disk_bytes"],
         "bounded opaque originals and receipts fit before any corpus access")
    directory.mkdir()
    partial = directory / "INCOMPLETE"
    partial.mkdir()
    write(directory / "STAGING_STARTED.json", dict(plan_sha256=sha(encoded(plan)), panel=plan["panel"],
        assessment_sha256=plan["assessment_sha256"], training_readback_sha256=plan["training_readback_sha256"],
        maximum_streams=len(keys), maximum_compressed_bytes=sum(plan["corpora"][k]["bytes"] for k in keys),
        status="OPAQUE_STAGING_CONSUMED_UNKNOWN", audio_headers_interpreted=0, native_decodes=0))
    started = clock()
    deadline = started + LIMITS["maximum_panel_seconds"]
    receipts, attempts, progress = {}, [], {}
    result = dict(schema="english-v6-b28-multicorpus-staging-result-v1", status="INCOMPLETE",
        plan_sha256=sha(encoded(plan)), http_attempts=attempts, current_stream=progress, panel=plan["panel"],
        assessment_sha256=plan["assessment_sha256"], training_readback_sha256=plan["training_readback_sha256"],
        original_compressed_archives_rehashed_offline=False, source_budget_authority_granted=False)
    try:
        for key in keys:
            need(clock() <= deadline, "complete-panel deadline before next stream")
            corpus = plan["corpora"][key]
            attempts.append(key)
            progress = {"archive_key": key, "compressed_bytes": 0, "opaque_selected_originals_staged": 0}
            result["current_stream"] = progress
            target = partial / key
            target.mkdir()
            with closing(open_response(corpus)) as response:
                validate_response(response, corpus)
                receipt = stage_archive(response, corpus, [r for r in plan["roster"] if r["archive_key"] == key],
                    catalogs[key], target, deadline, clock, progress,
                    LIMITS["selected_original_bytes_total"] - sum(x["selected_bytes_staged"] for x in receipts.values()))
            receipt["http"] = dict(url=corpus["url"], status=200, content_length=corpus["bytes"],
                                   ranges=0, redirects=0, retries=0)
            receipts[key] = receipt
            write(partial / (key + ".json"), receipt)
        need(sum(x["opaque_selected_originals_staged"] for x in receipts.values()) == 24
             and sum(x["selected_bytes_staged"] for x in receipts.values()) <= LIMITS["selected_original_bytes_total"],
             "one global24-original panel and global opaque-byte cap")
        release = dict(schema="english-v6-b28-whole-originals-release-v1", status="COMPLETE_VERIFIED",
            panel=plan["panel"], plan_sha256=sha(encoded(plan)), assessment_sha256=plan["assessment_sha256"],
            training_readback_sha256=plan["training_readback_sha256"], corpus_receipts=receipts,
            selected_original_slots=24, selected_speakers=12, opaque_members_staged=24,
            audio_headers_interpreted=0, native_decodes=0, tts_calls=0, observer_calls=0, model_calls=0,
            original_compressed_archives_retained=False, original_compressed_archives_rehashed_offline=False,
            source_budget_authority_granted=False)
        write(partial / "RELEASE.json", release)
        need(clock() <= deadline, "all corpus release checks inside fixed deadline")
        partial.rename(directory / "COMPLETE")
        result.update(status="COMPLETE_VERIFIED", release_sha256=sha(encoded(release)),
            compressed_bytes=sum(x["compressed_bytes"] for x in receipts.values()),
            opaque_members_staged=24, selected_bytes_staged=sum(x["selected_bytes_staged"] for x in receipts.values()))
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc)[:2048])
        raise
    finally:
        result["elapsed_seconds"] = clock() - started
        write(directory / "RESULT.json", result)
    return release


def read_release(directory, plan, catalogs):
    """Verify whole selected originals and recorded full-stream proof, no decode.

    Original producer/run/API and artifact authenticity remains an outer reader
    obligation. This does not claim to rehash the discarded compressed archives.
    """
    keys = validate_plan(plan, catalogs)
    directory = Path(directory)
    need(directory.name == "COMPLETE" and directory.is_dir() and not directory.is_symlink(), "only released complete staging")
    release_path = directory / "RELEASE.json"
    need(release_path.is_file() and not release_path.is_symlink() and release_path.stat().st_size <= 8388608, "bounded release proof")
    release = json.loads(release_path.read_bytes())
    need(release["schema"] == "english-v6-b28-whole-originals-release-v1" and release["status"] == "COMPLETE_VERIFIED"
         and release["plan_sha256"] == sha(encoded(plan)) and release["panel"] == plan["panel"]
         and release["assessment_sha256"] == plan["assessment_sha256"]
         and release["training_readback_sha256"] == plan["training_readback_sha256"]
         and sorted(release["corpus_receipts"]) == keys and release["selected_original_slots"] == 24
         and release["opaque_members_staged"] == 24 and release["selected_speakers"] == 12
         and all(release[k] == 0 for k in ("audio_headers_interpreted", "native_decodes", "tts_calls", "observer_calls", "model_calls"))
         and release["original_compressed_archives_retained"] is release["original_compressed_archives_rehashed_offline"]
             is release["source_budget_authority_granted"] is False, "same complete no-decode current-panel release")
    expected_files, by_id, total = {"RELEASE.json"}, {}, 0
    for key in keys:
        receipt = release["corpus_receipts"][key]
        corpus = plan["corpora"][key]
        need((directory / (key + ".json")).read_bytes() == encoded(receipt), "complete original corpus receipt bytes")
        expected_files.add(key + ".json")
        need(receipt["schema"] == "english-v6-b28-original-corpus-stream-v1" and receipt["status"] == "COMPLETE_VERIFIED"
             and receipt["corpus"] == corpus and receipt["compressed_bytes"] == corpus["bytes"]
             and receipt["compressed_sha256"] == corpus["sha256"] and receipt["compressed_md5"] == corpus["md5"]
             and receipt["gzip_crc_and_trailer_verified"] is receipt["compressed_eof_verified"]
                 is receipt["all_original_transcripts_verified"] is True
             and receipt["http"] == dict(url=corpus["url"], status=200, content_length=corpus["bytes"], ranges=0, redirects=0, retries=0),
             "exact original whole-stream verification and HTTP identity")
        fixed = {x["source_id"]: x for x in plan["roster"] if x["archive_key"] == key}
        need(len(receipt["selected_members"]) == len(fixed) == receipt["opaque_selected_originals_staged"], "all selected originals once")
        subtotal = 0
        for item in receipt["selected_members"]:
            ident = item["source_id"]
            need(ident in fixed and ident not in by_id and item["file"] == ident + ".flac"
                 and item["member"] == fixed[ident]["flac_member"], "exact planned original identity")
            path = directory / key / item["file"]
            need(path.is_file() and not path.is_symlink() and path.stat().st_size == item["bytes"] <= LIMITS["single_original_bytes"],
                 "whole original ordinary bounded file")
            with path.open("rb") as source:
                actual_sha = hashlib.file_digest(source, "sha256").hexdigest()
            pin = dict(bytes=item["bytes"], sha256=actual_sha)
            need(actual_sha == item["sha256"] and (fixed[ident]["original_flac"] is None or fixed[ident]["original_flac"] == pin),
                 "each whole retained original byte hash")
            subtotal += item["bytes"]
            by_id[ident] = dict(row=fixed[ident], path=path, flac_sha256=actual_sha, flac_bytes=item["bytes"],
                               source_group_id=source_group(fixed[ident]), id=natural_id(plan["panel"], fixed[ident]))
            expected_files.add(key + "/" + item["file"])
        need(subtotal == receipt["selected_bytes_staged"], "exact per-corpus selected byte total")
        total += subtotal
    need(total <= LIMITS["selected_original_bytes_total"] and len(by_id) == 24, "global exact retained originals and cap")
    actual = set()
    for path in directory.rglob("*"):
        need(not path.is_symlink(), "no symlink in released originals")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    need(actual == expected_files, "no omitted, extra, sealed-reserve or unrelated member")
    return [by_id[row["source_id"]] for row in plan["roster"]]
