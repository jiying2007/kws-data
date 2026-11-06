"""Bounded, immutable OpenSLR acoustic source subsets.

This is a research input receipt, not a second catalog or natural FAR dataset.
Every selected ZIP member is CRC checked. A partial retrieval never claims a
whole-archive checksum. The original WAV bytes and HTTP receipts are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import time
import urllib.request
import wave
import zipfile

SCHEMA = "kws-acoustic-source-subset-v1"
OFFICIAL_URL = "https://openslr.trmal.net/resources/28/rirs_noises.zip"
KINDS = {"measured-rir", "recorded-noise", "metadata"}
SPLITS = {"train", "development", "source-metadata"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode() + b"\n"


def sha(body):
    return hashlib.sha256(body).hexdigest()


def write_once(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(body)


def read_json(path):
    return json.loads(Path(path).read_bytes())


def validate_request(request):
    require(type(request) is dict and request.get("schema") == SCHEMA, "SOURCE_REQUEST_SCHEMA")
    source = request["source"]
    require(source["url"] == OFFICIAL_URL and source["homepage"] == "https://www.openslr.org/28/",
            "OFFICIAL_SLR28_SOURCE_REQUIRED")
    require(type(source["bytes"]) is int and 0 < source["bytes"] < 4 * 1024**3, "ARCHIVE_SIZE")
    require(type(source["etag"]) is str and source["etag"].startswith('"') and
            source["etag"].endswith('"'), "STRONG_ETAG_REQUIRED")
    require(request["purpose"] == "technical-research-only" and
            request["human_gold"] is False and request["accepted_natural_negative_hours"] == 0,
            "SOURCE_SCOPE")
    require(type(request["max_download_bytes"]) is int and
            0 < request["max_download_bytes"] <= 128 * 1024**2, "DOWNLOAD_BUDGET")
    require(type(request["max_elapsed_seconds"]) in (int, float) and
            0 < request["max_elapsed_seconds"] <= 900, "TIME_BUDGET")
    entries = request["members"]
    require(type(entries) is list and 1 <= len(entries) <= 32, "MEMBER_COUNT")
    seen, groups = set(), {}
    for row in entries:
        require(type(row) is dict and row["kind"] in KINDS and row["split"] in SPLITS,
                "MEMBER_KIND_OR_SPLIT")
        name = PurePosixPath(row["member"])
        require(not name.is_absolute() and ".." not in name.parts and
                name.parts[0] == "RIRS_NOISES" and not row["member"].endswith("/"), "MEMBER_PATH")
        require(row["member"] not in seen, "DUPLICATE_MEMBER")
        seen.add(row["member"])
        if row["kind"] == "metadata":
            require(row["split"] == "source-metadata", "METADATA_SPLIT")
        else:
            require(row["split"] in {"train", "development"} and row["member"].endswith(".wav"),
                    "AUDIO_SPLIT")
            group = row["source_group"]
            require(type(group) is str and bool(group), "SOURCE_GROUP_REQUIRED")
            require(groups.setdefault(group, row["split"]) == row["split"], "SOURCE_GROUP_CROSSES_SPLIT")
            if row["kind"] == "measured-rir":
                require("/real_rirs_isotropic_noises/" in row["member"], "MEASURED_RIR_MEMBER")
            else:
                require("/pointsource_noises/" in row["member"], "RECORDED_NOISE_MEMBER")
    return request


class RangeReader(io.RawIOBase):
    def __init__(self, source, output, max_bytes, max_seconds, opener=urllib.request.urlopen):
        self.source = source
        self.output = Path(output)
        self.max_bytes = max_bytes
        self.max_seconds = max_seconds
        self.opener = opener
        self.pos = 0
        self.total = 0
        self.started = time.monotonic()
        self.receipts = []

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        require(whence in (0, 1, 2), "SEEK_WHENCE")
        pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.source["bytes"] + offset
        require(0 <= pos <= self.source["bytes"], "SEEK_RANGE")
        self.pos = pos
        return pos

    def read(self, count=-1):
        if count < 0:
            count = self.source["bytes"] - self.pos
        count = min(count, self.source["bytes"] - self.pos)
        if not count:
            return b""
        elapsed = time.monotonic() - self.started
        require(elapsed < self.max_seconds, "DOWNLOAD_TIME_BUDGET")
        require(count <= 16 * 1024**2 and self.total + count <= self.max_bytes, "DOWNLOAD_BYTE_BUDGET")
        start, end = self.pos, self.pos + count - 1
        url = self.source["url"]
        request = urllib.request.Request(url, headers={
            "Range": f"bytes={start}-{end}", "If-Match": self.source["etag"],
            "Accept-Encoding": "identity"})
        with self.opener(request, timeout=min(40, self.max_seconds-elapsed)) as response:
            require(response.status == 206 and
                    response.headers.get("Content-Range") == f"bytes {start}-{end}/{self.source['bytes']}" and
                    response.headers.get("ETag") == self.source["etag"], "HTTP_RANGE_IDENTITY")
            body = response.read(count + 1)
            require(len(body) == count, "HTTP_RANGE_LENGTH")
            receipt = {"url": url, "status": response.status, "etag": self.source["etag"],
                       "start": start, "end": end, "archive_bytes": self.source["bytes"],
                       "bytes": len(body), "sha256": sha(body),
                       "response_headers": dict(response.headers)}
        index = len(self.receipts)
        write_once(self.output / f"range-{index:03}.bin", body)
        write_once(self.output / f"range-{index:03}.json", canonical(receipt))
        self.receipts.append(receipt)
        self.total += len(body)
        self.pos += len(body)
        return body


def wav_metadata(body):
    with wave.open(io.BytesIO(body), "rb") as reader:
        channels, rate, width, frames = (reader.getnchannels(), reader.getframerate(),
                                        reader.getsampwidth(), reader.getnframes())
        require(reader.getcomptype() == "NONE" and rate == 16000 and width == 2 and
                channels in (1, 2, 4, 8) and frames > 0, "WAV_FORMAT")
        raw = reader.readframes(frames)
        require(len(raw) == frames * channels * width, "WAV_TRUNCATED")
    return {"sample_rate_hz": rate, "channels": channels, "sample_width_bytes": width,
            "frames": frames, "duration_seconds": frames / rate, "pcm_sha256": sha(raw)}


def collect(request_path, output):
    request_path, output = Path(request_path), Path(output)
    request = validate_request(read_json(request_path))
    require(not output.exists(), "CREATE_ONLY_OUTPUT")
    output.mkdir(parents=True)
    write_once(output / "request.json", request_path.read_bytes())
    reader = RangeReader(request["source"], output / "http",
                         request["max_download_bytes"], request["max_elapsed_seconds"])
    records = []
    try:
        with zipfile.ZipFile(reader) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), "ZIP_DUPLICATE_MEMBER")
            for selected in request["members"]:
                info = archive.getinfo(selected["member"])
                require(0 < info.file_size <= 32 * 1024**2 and
                        info.compress_size <= 16 * 1024**2 and
                        not info.flag_bits & 1 and info.compress_type in (0, 8), "ZIP_MEMBER_BOUNDS")
                body = archive.read(info)
                require(len(body) == info.file_size, "ZIP_MEMBER_SIZE")
                destination = output / "original" / selected["member"]
                write_once(destination, body)
                record = {**selected, "relative_path": destination.relative_to(output).as_posix(),
                          "bytes": len(body), "sha256": sha(body),
                          "zip_crc32": f"{info.CRC:08x}", "zip_crc_verified": True,
                          "zip_header_offset": info.header_offset,
                          "license_basis": request["source"]["license_basis"],
                          "human_gold": False, "keyword_content_verified": False,
                          "training_admitted": False}
                if selected["kind"] != "metadata":
                    record["audio"] = wav_metadata(body)
                records.append(record)
        summary = {"schema": SCHEMA, "request_sha256": sha(request_path.read_bytes()),
                   "source": request["source"], "assets": records,
                   "http_range_reads": len(reader.receipts), "downloaded_bytes": reader.total,
                   "elapsed_seconds": time.monotonic() - reader.started,
                   "whole_archive_downloaded": False, "whole_archive_crc_verified": False,
                   "original_audio_bytes_preserved": True, "model_calls": 0,
                   "accepted_natural_negative_hours": 0, "human_gold": False,
                   "scene_distances_measured_by_project": False}
        write_once(output / "source-subset.json", canonical(summary))
        files = {}
        for path in sorted(output.rglob("*")):
            if path.is_file():
                body = path.read_bytes()
                files[path.relative_to(output).as_posix()] = {"sha256": sha(body), "bytes": len(body)}
        write_once(output / "MANIFEST.json", canonical({"schema": SCHEMA, "files": files}))
        return summary
    except BaseException as exc:
        write_once(output / "failed.json", canonical({"error": type(exc).__name__,
                   "message": str(exc), "downloaded_bytes": reader.total,
                   "model_calls": 0, "partial_outputs_preserved": True}))
        raise


def verify(root):
    root = Path(root)
    manifest = read_json(root / "MANIFEST.json")
    require(manifest["schema"] == SCHEMA, "MANIFEST_SCHEMA")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == set(manifest["files"]) | {"MANIFEST.json"}, "MEMBER_ROSTER_CHANGED")
    for name, expected in manifest["files"].items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts, "MANIFEST_PATH")
        body = (root / name).read_bytes()
        require(sha(body) == expected["sha256"] and len(body) == expected["bytes"], "MEMBER_HASH:" + name)
    request = validate_request(read_json(root / "request.json"))
    summary = read_json(root / "source-subset.json")
    require(summary["request_sha256"] == sha((root / "request.json").read_bytes()) and
            summary["source"] == request["source"] and
            summary["whole_archive_crc_verified"] is False and summary["model_calls"] == 0,
            "SUBSET_SUMMARY")
    require(len(summary["assets"]) == len(request["members"]), "ASSET_COUNT")
    for selected, asset in zip(request["members"], summary["assets"]):
        require(all(asset[k] == v for k, v in selected.items()), "ASSET_REQUEST_BINDING")
        body = (root / asset["relative_path"]).read_bytes()
        require(sha(body) == asset["sha256"] and len(body) == asset["bytes"], "ASSET_HASH")
        if selected["kind"] != "metadata":
            require(asset["audio"] == wav_metadata(body), "AUDIO_METADATA")
    return {"status": "PASS", "files_verified": len(manifest["files"]),
            "assets": len(summary["assets"]), "model_calls": 0}


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("collect")
    command.add_argument("--request", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = sub.add_parser("verify")
    command.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = collect(args.request, args.output) if args.command == "collect" else verify(args.root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
