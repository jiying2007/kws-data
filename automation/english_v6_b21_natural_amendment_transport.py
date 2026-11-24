"""New amendment authorization around unchanged fixed-mirror byte transport."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import threading
import time
import urllib.request
import english_v6_b21_natural_amendment_source_ancestry as lineage
from english_v6_b21_fixed_mirror_transport import (
    PLAN, plan, ranges, transfer, OfficialRedirect, verify_transport, read_receipt)
need, digest, record = lineage.need, lineage.digest, lineage.qa.record

def download(corpus, directory, prepared_parent):
    identity = lineage.execution_guard(prepared_parent)
    permission = lineage.require_authorization(directory)
    need(permission['source_execution'] == identity, 'same audited continuation before transport')
    lineage.one_execution(identity, False)
    spec = plan()
    corpus, directory = (Path(corpus), Path(directory))
    partial = corpus.with_name(corpus.name + '.partial')
    transport = directory / 'TRANSPORT'
    need(corpus.is_absolute() and corpus.parent.is_dir() and (corpus.name == 'train-clean-100.tar.gz') and (not corpus.exists()) and (not partial.exists()) and (not transport.exists()), 'one fresh corpus transport, no resume or retry')
    need(os.statvfs(corpus.parent).f_bavail * os.statvfs(corpus.parent).f_frsize > spec['bytes'] + 536870912, 'one complete archive plus bounded working space, no duplicate range files')
    transport.mkdir()
    receipts = transport / 'ranges'
    receipts.mkdir()
    parts = ranges(spec['bytes'], spec['parallel_ranges'])
    record(transport / 'STARTED.json', dict(source_execution=identity, transport_plan_sha256=digest(PLAN.read_bytes()), transport_origin=permission['transport_origin'], ranges=parts, range_retries=0, source_calls=0, model_calls=0))
    started, stop = (time.monotonic(), threading.Event())
    deadline = started + spec['maximum_elapsed_seconds']
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 384)
    report = dict(schema=spec['schema'], status='STOP', urls=spec['urls'], bytes=spec['bytes'], sha256=spec['sha256'], transport_plan_sha256=digest(PLAN.read_bytes()), ranges=[], range_retries=0, source_calls=0, model_calls=0)
    try:
        os.ftruncate(fd, spec['bytes'])
        with ThreadPoolExecutor(max_workers=spec['parallel_ranges']) as executor:
            futures = [executor.submit(transfer, part, spec, fd, receipts, stop, deadline, urllib.request.build_opener(OfficialRedirect(spec['urls'][0]))) for part in parts]
            try:
                for future in as_completed(futures, timeout=spec['maximum_elapsed_seconds']):
                    report['ranges'].append(future.result())
            except BaseException:
                stop.set()
                raise
        os.fsync(fd)
        need(time.monotonic() <= deadline and partial.stat().st_size == spec['bytes'], 'bounded exact whole archive before hashing')
        with partial.open('rb') as stream:
            need(hashlib.file_digest(stream, 'sha256').hexdigest() == spec['sha256'], 'every original compressed byte matches frozen corpus')
        need(time.monotonic() <= deadline and (not corpus.exists()), 'whole verification completed in the fixed deadline')
        partial.rename(corpus)
        report.update(status='COMPLETE_VERIFIED', ranges=sorted(report['ranges'], key=lambda part: part['index']))
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        os.close(fd)
        report['elapsed_seconds'] = time.monotonic() - started
        record(transport / 'RESULT.json', report)
    record(directory / 'CORPUS_TRANSPORT.json', report)
    return verify_transport(corpus, directory)
