"""Bounded transport of the identical official corpus; no archive/audio parsing.

Eight nonoverlapping GET ranges write one exclusive partial file directly.
Only a complete whole-file length and SHA match permits an atomic rename.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import io
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
import zipfile

import english_v6_b21_fixed_mirror_source_ancestry as lineage

need, digest, record = lineage.need, lineage.digest, lineage.qa.record
PLAN = Path(__file__).with_name("english_v6_b21_fixed_mirror_transport.json")
PROBE = Path(__file__).with_name("english_v6_b21_continuation_probe")
OFFICIAL = (
    "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "https://openslr.trmal.net/resources/12/train-clean-100.tar.gz",
    "https://openslr.elda.org/resources/12/train-clean-100.tar.gz",
    "https://openslr.magicdatatech.com/resources/12/train-clean-100.tar.gz",
)


def plan(require_bound=True):
    value = json.loads(PLAN.read_bytes())
    need(value == dict(schema="english-v6-b21-fixed-mirror-corpus-range-transport-v1", urls=[OFFICIAL[3]],
        bytes=lineage.CORPUS_BYTES, sha256=lineage.CORPUS_SHA, parallel_ranges=8, maximum_elapsed_seconds=2700,
        socket_timeout_seconds=60, maximum_chunk_bytes=1048576, range_retries=0, fallback_urls=[],
        archive_parsing_during_transport=False, source_or_model_calls_during_transport=0,
        range_url_selection="fixed_magicdatatech", maximum_redirects_per_range=4,
        failed_transport_run_id=45365292922,
        redirect_policy="magicdatatech_https_huggingface_only", probe=value["probe"]),
        "one fixed corpus, eight bounded transport ranges and no source operations")
    verify_probe(value["probe"])
    need(lineage.origin.verify()["failed_continuation_run_id"] == value["failed_transport_run_id"],
         "actual prior completed magicdatatech ranges and zero scientific calls")
    return value


def verify_probe(proof):
    pin = dict(id=14903391010, run_id=45357232725, run_head="3e322a31292bf08b8ba5f4489eccc20d58b9141b",
        bytes=1824, sha256="423246f00626cd1d5ef051e569a2935809e9b28cd19589674d74fa65f503c92e")
    need(proof["artifact"] == pin and set(proof["files"]) == {"original.zip", "api/run.json", "api/artifact.json", "original/RESULT.json"},
         "actual original bounded official-mirror probe, never a synthetic speed claim")
    for name, expected in proof["files"].items():
        raw = (PROBE / name).read_bytes()
        need(len(raw) == expected["bytes"] and digest(raw) == expected["sha256"], "original transport probe bytes " + name)
    raw = (PROBE / "original.zip").read_bytes()
    need(len(raw) == pin["bytes"] and digest(raw) == pin["sha256"], "exact original probe ZIP")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        need(archive.namelist() == ["RESULT.json"] and archive.testzip() is None
             and archive.read("RESULT.json") == (PROBE / "original/RESULT.json").read_bytes(), "exact original probe result member")
    report = json.loads((PROBE / "original/RESULT.json").read_bytes())
    run = json.loads((PROBE / "api/run.json").read_bytes()); artifact = json.loads((PROBE / "api/artifact.json").read_bytes())
    need(run["id"] == pin["run_id"] and run["head_sha"] == pin["run_head"] and run["run_attempt"] == 1
         and run["status"] == "completed" and run["conclusion"] == "success"
         and run["repository"]["full_name"] == lineage.REPO and run["repository"]["private"] is False
         and artifact["id"] == pin["id"] and artifact["size_in_bytes"] == pin["bytes"]
         and artifact["digest"] == "sha256:" + pin["sha256"]
         and artifact["workflow_run"]["id"] == pin["run_id"] and artifact["workflow_run"]["head_sha"] == pin["run_head"],
         "same actual public original transport probe and archive identity")
    need(report["schema"] == "english-v6-b21-bounded-corpus-transport-probe-v1"
         and report["run_id"] == str(pin["run_id"]) and report["run_head"] == pin["run_head"]
         and report["received_bytes"] == report["maximum_received_bytes"] == 16777216
         and report["whole_archive_bytes"] == lineage.CORPUS_BYTES and report["whole_archive_verified"] is False
         and report["valid_mirrors"] == 4 and report["valid_prefixes_agree"] is True
         and all(report[k] == 0 for k in ("archive_entries_parsed", "audio_decodes", "tts_calls", "observer_calls", "kws_forwards", "optimizer_updates"))
         and [entry["url"] for entry in report["reports"]] == list(OFFICIAL), "metadata-only four official ranges, no source generation or decoding")
    for entry in report["reports"]:
        need(entry["status"] == 206 and entry["valid_range"] is True and entry["received"] == 4194304
             and entry["content_length"] == "4194304" and entry["content_range"] == "bytes 0-4194303/6387309499"
             and entry["prefix_sha256"] == "ca9173d848c83fe81b1de64bf7a1720df5d485afad2b6ee2078991de22dc5802",
             "all observed prefixes match while full-file verification remains mandatory")
    return report


def ranges(total, count):
    need(type(total) is int and type(count) is int and total >= count > 0, "positive bounded complete range partition")
    return [dict(index=i, start=total * i // count, end=total * (i + 1) // count - 1) for i in range(count)]


def validate_headers(status, headers, part, total):
    expected = "bytes " + str(part["start"]) + "-" + str(part["end"]) + "/" + str(total)
    need(status == 206 and headers.get("Content-Range") == expected
         and headers.get("Content-Length") == str(part["end"] - part["start"] + 1)
         and headers.get("Content-Encoding", "identity").lower() == "identity",
         "exact206 native byte range, whole corpus length and no encoded substitution")


def final_url_allowed(original, final):
    if final == original:
        return True
    parsed = urllib.parse.urlsplit(final)
    host = parsed.hostname or ""
    return original == OFFICIAL[3] and parsed.scheme == "https" and (host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".hf.co"))


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    max_redirections = 4
    max_repeats = 1

    def __init__(self, original):
        self.original = original

    def redirect_request(self, request, fp, code, message, headers, newurl):
        need(final_url_allowed(self.original, newurl), "only the observed official magicdatatech HTTPS/HuggingFace redirect chain")
        redirected = super().redirect_request(request, fp, code, message, headers, newurl)
        need(redirected is not None and redirected.get_method() == "GET"
             and redirected.get_header("Range") == request.get_header("Range"), "same exact registered Range across provider redirect")
        return redirected


def transfer(part, spec, fd, receipts, stop, deadline, opener):
    index, start, end = part["index"], part["start"], part["end"]
    url = spec["urls"][0]
    record(receipts / (str(index) + "-STARTED.json"), dict(part, url=url, attempt=1))
    request = urllib.request.Request(url, headers={
        "Range": "bytes=" + str(start) + "-" + str(end), "Accept-Encoding": "identity",
        "User-Agent": "kws-b21-audited-corpus-transport/1"}, method="GET")
    total, received, checksum = spec["bytes"], 0, hashlib.sha256()
    need(not stop.is_set() and time.monotonic() < deadline, "transport deadline before range entry")
    timeout = min(spec["socket_timeout_seconds"], max(.1, deadline - time.monotonic()))
    with opener.open(request, timeout=timeout) as response:
        need(final_url_allowed(url, response.geturl()), "stable official mirror and observed bounded provider redirect")
        final_url = response.geturl()
        validate_headers(response.status, response.headers, part, total)
        while received < end - start + 1:
            need(not stop.is_set() and time.monotonic() < deadline, "bounded transport deadline or previous range failure")
            chunk = response.read(min(spec["maximum_chunk_bytes"], end - start + 1 - received))
            need(bool(chunk), "no short range or silent truncation")
            offset = 0
            while offset < len(chunk):
                written = os.pwrite(fd, chunk[offset:], start + received + offset)
                need(written > 0, "complete direct partial-file write")
                offset += written
            checksum.update(chunk); received += len(chunk)
        need(response.read(1) == b"" and time.monotonic() <= deadline, "exact complete range within bounded deadline")
    value = dict(part, url=url, final_url=final_url, attempt=1, received_bytes=received,
        content_range="bytes " + str(start) + "-" + str(end) + "/" + str(total),
        sha256=checksum.hexdigest(), status="COMPLETE")
    record(receipts / (str(index) + "-RETURNED.json"), value)
    return value


def read_receipt(directory):
    directory = Path(directory)
    spec = plan()
    result = json.loads((directory / "CORPUS_TRANSPORT.json").read_bytes())
    expected_parts = ranges(spec["bytes"], spec["parallel_ranges"])
    need(result["schema"] == spec["schema"] and result["status"] == "COMPLETE_VERIFIED"
         and result["transport_plan_sha256"] == digest(PLAN.read_bytes())
         and result["urls"] == spec["urls"] and result["bytes"] == spec["bytes"]
         and result["sha256"] == spec["sha256"]
         and type(result["elapsed_seconds"]) in (int, float) and 0 <= result["elapsed_seconds"] <= spec["maximum_elapsed_seconds"]
         and result["range_retries"] == result["source_calls"] == result["model_calls"] == 0
         and len(result["ranges"]) == spec["parallel_ranges"], "all transport completed before source access")
    for part, returned in zip(expected_parts, result["ranges"], strict=True):
        prefix = directory / "TRANSPORT" / "ranges" / str(part["index"])
        started = json.loads(Path(str(prefix) + "-STARTED.json").read_bytes())
        url = spec["urls"][0]
        need(started == dict(part, url=url, attempt=1)
             and returned == json.loads(Path(str(prefix) + "-RETURNED.json").read_bytes())
             and all(returned[k] == v for k, v in part.items())
             and returned["status"] == "COMPLETE" and returned["attempt"] == 1
             and returned["url"] == url and final_url_allowed(url, returned["final_url"])
             and returned["received_bytes"] == part["end"] - part["start"] + 1
             and returned["content_range"] == "bytes " + str(part["start"]) + "-" + str(part["end"]) + "/" + str(spec["bytes"])
             and re.fullmatch(r"[0-9a-f]{64}", returned["sha256"] or "") is not None,
             "every disjoint requested range returned its complete exact bytes once")
    need(json.loads((directory / "TRANSPORT/RESULT.json").read_bytes()) == result,
         "same final transport result retained before the generation marker")
    return result


def verify_transport(corpus, directory):
    result = read_receipt(directory)
    corpus = Path(corpus)
    need(corpus.is_file() and not corpus.is_symlink() and corpus.stat().st_size == result["bytes"],
         "complete original whole archive, never a partial transport")
    with corpus.open("rb") as stream:
        need(hashlib.file_digest(stream, "sha256").hexdigest() == result["sha256"], "original whole-corpus hash before generation")
    return result


def download(corpus, directory, prepared_parent):
    identity = lineage.execution_guard(prepared_parent)
    permission = lineage.require_authorization(directory)
    need(permission["source_execution"] == identity, "same audited continuation before transport")
    lineage.one_execution(identity, False)
    spec = plan()
    corpus, directory = Path(corpus), Path(directory)
    partial = corpus.with_name(corpus.name + ".partial")
    transport = directory / "TRANSPORT"
    need(corpus.is_absolute() and corpus.parent.is_dir() and corpus.name == "train-clean-100.tar.gz"
         and not corpus.exists() and not partial.exists() and not transport.exists(), "one fresh corpus transport, no resume or retry")
    need(os.statvfs(corpus.parent).f_bavail * os.statvfs(corpus.parent).f_frsize > spec["bytes"] + 536870912,
         "one complete archive plus bounded working space, no duplicate range files")
    transport.mkdir(); receipts = transport / "ranges"; receipts.mkdir()
    parts = ranges(spec["bytes"], spec["parallel_ranges"])
    record(transport / "STARTED.json", dict(source_execution=identity, transport_plan_sha256=digest(PLAN.read_bytes()),
        transport_origin=permission["transport_origin"], ranges=parts, range_retries=0, source_calls=0, model_calls=0))
    started, stop = time.monotonic(), threading.Event()
    deadline = started + spec["maximum_elapsed_seconds"]
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    report = dict(schema=spec["schema"], status="STOP", urls=spec["urls"], bytes=spec["bytes"], sha256=spec["sha256"],
        transport_plan_sha256=digest(PLAN.read_bytes()), ranges=[], range_retries=0, source_calls=0, model_calls=0)
    try:
        os.ftruncate(fd, spec["bytes"])
        with ThreadPoolExecutor(max_workers=spec["parallel_ranges"]) as executor:
            futures = [executor.submit(transfer, part, spec, fd, receipts, stop, deadline,
                       urllib.request.build_opener(OfficialRedirect(spec["urls"][0]))) for part in parts]
            try:
                for future in as_completed(futures, timeout=spec["maximum_elapsed_seconds"]):
                    report["ranges"].append(future.result())
            except BaseException:
                stop.set()
                raise
        os.fsync(fd)
        need(time.monotonic() <= deadline and partial.stat().st_size == spec["bytes"], "bounded exact whole archive before hashing")
        with partial.open("rb") as stream:
            need(hashlib.file_digest(stream, "sha256").hexdigest() == spec["sha256"], "every original compressed byte matches frozen corpus")
        need(time.monotonic() <= deadline and not corpus.exists(), "whole verification completed in the fixed deadline")
        partial.rename(corpus)
        report.update(status="COMPLETE_VERIFIED", ranges=sorted(report["ranges"], key=lambda part: part["index"]))
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        os.close(fd)
        report["elapsed_seconds"] = time.monotonic() - started
        record(transport / "RESULT.json", report)
    record(directory / "CORPUS_TRANSPORT.json", report)
    return verify_transport(corpus, directory)
