#!/usr/bin/env python3
"""Record each actual URL-open attempt while reusing the frozen AISHELL fetcher."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import fetch_aishell_subset as fetcher

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def put(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--etag", required=True)
    p.add_argument("--bytes", type=int, required=True)
    p.add_argument("--attempts", type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.bytes <= 128 * 1024 * 1024:
        raise ValueError("THIS_TRANCHE_MAXIMUM_128_MIB")
    a.attempts.mkdir(parents=False, exist_ok=False)
    opener = urllib.request.build_opener(NoRedirect())
    count = 0
    def observed_open(request, timeout):
        nonlocal count
        count += 1
        item = a.attempts / f"attempt-{count:04d}"
        item.mkdir(exist_ok=False)
        claim = {"schema": "kws-observed-http-attempt-v1", "attempt_ordinal": count,
                 "started_at": utc(), "url": request.full_url, "method": request.get_method(),
                 "request_headers": dict(request.header_items()), "timeout_seconds": timeout,
                 "boundary": "Written immediately before urllib.request.urlopen; no response is assumed."}
        put(item / "request.json", claim)
        began = time.monotonic_ns()
        try:
            response = opener.open(request, timeout=timeout)
        except Exception as error:
            put(item / "response.json", {"finished_at": utc(), "elapsed_ns": time.monotonic_ns() - began,
                "status": "raised", "exception_type": type(error).__name__,
                "http_status": getattr(error, "code", None), "error": str(error)})
            raise
        put(item / "response.json", {"finished_at": utc(), "elapsed_ns": time.monotonic_ns() - began,
            "status": "headers_returned_body_not_yet_verified", "http_status": response.status,
            "response_url": response.geturl(), "redirects_allowed": False,
            "response_headers": {key: response.headers.get(key) for key in
                                 ("ETag", "Content-Range", "Content-Length", "Content-Encoding")},
            "boundary": "Exact body bytes/hash and Range acceptance remain the fetcher's range receipt."})
        return response
    result = fetcher.fetch_prefix(a.output, etag=a.etag, target_bytes=a.bytes,
                                  opener=observed_open)
    result_view = {k: v for k, v in result.items() if k != "retrieved_ranges"}
    result_view["urlopen_attempts_observed"] = count
    result_view["wrapper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(result_view, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

