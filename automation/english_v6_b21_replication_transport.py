"""Transport the fixed replication corpus after an original primary PASS.

The fixed-mirror primary downloader stays byte-for-byte frozen. This adapter
reuses its exact range worker, mirror proof and whole-file verification while
binding every receipt to the separately authorized replication source event.
It performs no archive parsing, audio generation, observation or KWS inference.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import english_v6_b21_fixed_mirror_transport as transport
import english_v6_b21_replication_source_ancestry as lineage

SHARED_TRANSPORT_SHA = "eca61c7239562aac0fdb08ef5bf93dec93a017dcf771f78e19b857c0b4d35db4"
need, digest, record = lineage.need, lineage.digest, lineage.qa.record


def plan():
    need(digest(Path(transport.__file__).read_bytes()) == SHARED_TRANSPORT_SHA,
         "unchanged reviewed whole-corpus range transport implementation")
    spec = transport.plan()
    upstream = lineage.read_plan()["replication"]["natural"]["upstream"]
    need(spec["bytes"] == upstream["bytes"] and spec["sha256"] == upstream["sha256"],
         "transport cannot change the fixed replication corpus")
    return spec


def authorization(directory, prepared_parent):
    identity = lineage.execution_guard(prepared_parent)
    permission = lineage.require_authorization(directory)
    need(permission["source_execution"] == identity,
         "original primary PASS and this exact first replication event before transport")
    return identity, permission


def receipt_identity(identity, permission):
    return dict(panel="replication", source_execution=identity,
        primary_quality_lock_sha256=permission["primary_quality_lock_sha256"],
        transport_adapter_sha256=digest(Path(__file__).read_bytes()),
        shared_transport_sha256=SHARED_TRANSPORT_SHA)


def verify(corpus, directory, prepared_parent):
    identity, permission = authorization(directory, prepared_parent)
    spec = plan()
    directory = Path(directory)
    result = transport.verify_transport(corpus, directory)
    binding = receipt_identity(identity, permission)
    need(all(result.get(key) == value for key, value in binding.items()),
         "complete verified corpus belongs to this authorized replication event")
    started = json.loads((directory / "TRANSPORT/STARTED.json").read_bytes())
    need(started == dict(binding, transport_plan_sha256=digest(transport.PLAN.read_bytes()),
        ranges=transport.ranges(spec["bytes"], spec["parallel_ranges"]),
        range_retries=0, source_calls=0, model_calls=0),
        "exact before-request replication transport receipt and original PASS lock")
    return result


def download(corpus, directory, prepared_parent):
    identity, permission = authorization(directory, prepared_parent)
    lineage.require_uploaded_opening(directory, prepared_parent)
    spec = plan()
    corpus, directory = Path(corpus), Path(directory)
    partial = corpus.with_name(corpus.name + ".partial")
    retained = directory / "TRANSPORT"
    need(corpus.is_absolute() and corpus.parent.is_dir() and corpus.name == "train-clean-100.tar.gz"
         and not corpus.exists() and not partial.exists() and not retained.exists()
         and not (directory / "CORPUS_TRANSPORT.json").exists(),
         "one fresh replication corpus transport, no resume or retry")
    need(os.statvfs(corpus.parent).f_bavail * os.statvfs(corpus.parent).f_frsize > spec["bytes"] + 536870912,
         "one complete archive plus bounded working space, no duplicate range files")
    retained.mkdir(); receipts = retained / "ranges"; receipts.mkdir()
    parts = transport.ranges(spec["bytes"], spec["parallel_ranges"])
    binding = receipt_identity(identity, permission)
    record(retained / "STARTED.json", dict(binding, transport_plan_sha256=digest(transport.PLAN.read_bytes()),
        ranges=parts, range_retries=0, source_calls=0, model_calls=0))
    started, stop = time.monotonic(), threading.Event()
    deadline = started + spec["maximum_elapsed_seconds"]
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    report = dict(binding, schema=spec["schema"], status="STOP", urls=spec["urls"],
        bytes=spec["bytes"], sha256=spec["sha256"],
        transport_plan_sha256=digest(transport.PLAN.read_bytes()), ranges=[],
        range_retries=0, source_calls=0, model_calls=0)
    try:
        os.ftruncate(fd, spec["bytes"])
        with ThreadPoolExecutor(max_workers=spec["parallel_ranges"]) as executor:
            futures = [executor.submit(transport.transfer, part, spec, fd, receipts, stop, deadline,
                transport.urllib.request.build_opener(transport.OfficialRedirect(spec["urls"][0])))
                for part in parts]
            try:
                for future in as_completed(futures, timeout=spec["maximum_elapsed_seconds"]):
                    report["ranges"].append(future.result())
            except BaseException:
                stop.set()
                raise
        os.fsync(fd)
        need(time.monotonic() <= deadline and partial.stat().st_size == spec["bytes"],
             "bounded exact whole archive before hashing")
        with partial.open("rb") as stream:
            need(hashlib.file_digest(stream, "sha256").hexdigest() == spec["sha256"],
                 "every original compressed byte matches the fixed replication corpus")
        need(time.monotonic() <= deadline and not corpus.exists(),
             "whole verification completed in the fixed deadline")
        partial.rename(corpus)
        report.update(status="COMPLETE_VERIFIED", ranges=sorted(report["ranges"], key=lambda part: part["index"]))
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        os.close(fd)
        report["elapsed_seconds"] = time.monotonic() - started
        record(retained / "RESULT.json", report)
    record(directory / "CORPUS_TRANSPORT.json", report)
    return verify(corpus, directory, prepared_parent)


def check():
    opening = lineage.read_opening(require_bound=False)
    spec = plan()
    status = ("INERT_REQUIRES_ORIGINAL_PRIMARY_PASS" if opening["primary_quality_lock"] is None
              else "BOUND_REQUIRES_ORIGINAL_PRIMARY_PASS_READBACK")
    return dict(status=status, bytes=spec["bytes"], sha256=spec["sha256"],
        parallel_ranges=spec["parallel_ranges"], source_calls=0, model_calls=0, network_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "download", "verify"))
    parser.add_argument("corpus", nargs="?", type=Path)
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("prepared_parent", nargs="?")
    args = parser.parse_args()
    if args.mode == "check":
        print(json.dumps(check(), sort_keys=True))
    else:
        need(args.corpus is not None and args.directory is not None and args.prepared_parent is not None,
             "exact corpus destination, saved authorization and reviewed execution parent")
        result = (download if args.mode == "download" else verify)(args.corpus, args.directory, args.prepared_parent)
        print(json.dumps(result, sort_keys=True))
