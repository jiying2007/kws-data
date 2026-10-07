#!/usr/bin/env python3
"""Offline, content-addressed research archive. Never executes stored content.

Only Python's standard library is needed. See ARCHIVE_FORMAT.md for the public
format, limits, provenance responsibilities, and examples. This is an integrity
checker, not a signature verifier or a rights-clearance tool.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import wave
import zlib

SCHEMA = "bounded-research-archive-v1"
LIMIT = 65536
MAX_OBJECT = 128 * 1024 * 1024
MAX_TOTAL = 512 * 1024 * 1024
MAX_ASSETS = 10000
HEX = re.compile(r"[0-9a-f]{64}\Z")


class ArchiveError(ValueError):
    pass


def require(test, message):
    if not test:
        raise ArchiveError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def safe_name(value):
    require(isinstance(value, str) and value and "\\" not in value and
            "\x00" not in value and not value.startswith("/"), "unsafe relative path")
    require(all(p not in ("", ".", "..") for p in value.split("/")),
            "unsafe relative path")
    require(not re.search(r'[:<>"|?*\x00-\x1f]', value), "nonportable path characters")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        prefix + n for prefix in ("COM", "LPT") for n in "123456789¹²³"}
    require(all(not p.endswith((" ", ".")) and p.split(".")[0].upper() not in reserved
                for p in value.split("/")), "nonportable path component")
    require(not PurePosixPath(value).is_absolute(), "absolute path")
    return value


def check_hash(value):
    require(isinstance(value, str) and HEX.fullmatch(value), "invalid SHA-256")
    return value


def check_size(value, limit=MAX_OBJECT):
    require(type(value) is int and 0 <= value <= limit, "invalid or excessive byte size")
    return value


def check_root(root):
    root = Path(root)
    absolute = root.absolute()
    require(not any(p.is_symlink() for p in (absolute, *absolute.parents)),
            "symlink in archive/output root ancestry")
    return root


def safe_path(root, name):
    root = check_root(root)
    name = safe_name(name)
    path = root
    for part in name.split("/"):
        path = path / part
        require(not path.is_symlink(), "symlink in archive/output path")
    return path


def write_identical(root, name, data):
    path = safe_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file() and path.read_bytes() == data,
                "refusing to replace nonidentical artifact: " + name)
    else:
        with path.open("xb") as f:
            f.write(data)
    return {"path": name, "sha256": digest(data), "size_bytes": len(data)}


def read_checked(root, record, limit=LIMIT):
    size = check_size(record["size_bytes"], limit)
    sha = check_hash(record["sha256"])
    path = safe_path(root, record["path"])
    require(path.is_file() and path.stat().st_size == size, "missing file or size mismatch: " + record["path"])
    with path.open("rb") as f:
        data = f.read(size + 1)
    require(len(data) == size and digest(data) == sha, "SHA-256 mismatch: " + record["path"])
    return data


def deterministic_gzip(data):
    out = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=6,
                       fileobj=out, mtime=0) as f:
        f.write(data)
    return out.getvalue()


def unpack_gzip(data, expected_size):
    check_size(expected_size)
    decoder = zlib.decompressobj(wbits=31)
    try:
        result = decoder.decompress(data, expected_size + 1)
        require(len(result) == expected_size, "decompressed length mismatch")
        require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
                "incomplete, trailing, concatenated, or oversized gzip data")
        return result
    except zlib.error as exc:
        raise ArchiveError("invalid gzip stream") from exc


def wav_pcm(data):
    try:
        with wave.open(io.BytesIO(data), "rb") as f:
            require((f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getcomptype())
                    == (1, 2, 16000, "NONE"), "WAV must be mono PCM16LE at 16000 Hz")
            count = f.getnframes()
            pcm = f.readframes(count + 1)
            require(len(pcm) == count * 2, "WAV sample count mismatch")
            return pcm
    except (wave.Error, EOFError) as exc:
        raise ArchiveError("invalid PCM WAV") from exc


def public_entry(entry):
    require(isinstance(entry, dict), "asset must be an object")
    asset = {k: v for k, v in entry.items() if k != "source_path"}
    safe_name(asset["logical_path"])
    check_hash(asset["sha256"])
    check_size(asset["size_bytes"])
    if "source_logical_path" in asset:
        safe_name(asset["source_logical_path"])
        require(asset.get("transformation") == "wav-pcm-s16le", "unsupported derivation")
    return asset


def page_records(root, kind, entries):
    """Bound the encoded size, including JSON punctuation and UTF-8 expansion."""
    records, page = [], []
    for entry in entries:
        candidate = page + [entry]
        if len(canonical({kind: candidate})) > LIMIT:
            require(page, "single metadata entry exceeds shard limit")
            raw = canonical({kind: page})
            records.append(write_identical(root, "archive/" + kind + "/" + digest(raw) + ".json", raw))
            page = [entry]
            require(len(canonical({kind: page})) <= LIMIT, "single metadata entry exceeds shard limit")
        else:
            page = candidate
    if page:
        raw = canonical({kind: page})
        records.append(write_identical(root, "archive/" + kind + "/" + digest(raw) + ".json", raw))
    return records


def pack(plan_path, root):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    entries = plan if isinstance(plan, list) else plan["assets"]
    require(isinstance(entries, list) and len(entries) <= MAX_ASSETS, "invalid asset count")
    root = check_root(root)
    root.mkdir(parents=True, exist_ok=True)
    assets = [public_entry(e) for e in entries]
    by_name = {a["logical_path"]: a for a in assets}
    require(len(by_name) == len(assets), "duplicate logical path")
    # Reject ancestor/file collisions before writing any materializable payload.
    for name in by_name:
        require(not any(str(p) in by_name for p in PurePosixPath(name).parents if str(p) != "."),
                "logical file/directory collision")
    require(sum(a["size_bytes"] for a in assets) <= MAX_TOTAL,
            "archive logical bytes exceed safety limit")
    objects, object_data, shards = {}, {}, {}
    for entry, asset in zip(entries, assets):
        if "source_logical_path" in asset:
            require("source_path" not in entry, "derived asset must not duplicate source bytes")
            continue
        path = Path(entry["source_path"])
        require(path.is_file() and not path.is_symlink(), "source must be a regular nonsymlink file")
        require(path.stat().st_size == asset["size_bytes"], "source size mismatch")
        data = path.read_bytes()
        sha = asset["sha256"]
        require(digest(data) == sha, "source SHA-256 mismatch: " + asset["logical_path"])
        if sha in objects:
            require(object_data[sha] == data, "digest collision")
            continue
        object_data[sha] = data
        compressed = deterministic_gzip(data)
        parts = []
        for offset in range(0, len(compressed), LIMIT):
            block = compressed[offset:offset + LIMIT]
            record = write_identical(root, "archive/shards/" + digest(block) + ".part", block)
            parts.append(record)
            shards[record["path"]] = record["size_bytes"]
        desc = {"sha256": sha, "size_bytes": len(data), "compression": "gzip",
                "compressed_sha256": digest(compressed), "compressed_size_bytes": len(compressed),
                "shards": parts}
        raw = canonical(desc)
        require(len(raw) <= LIMIT, "object descriptor exceeds limit")
        objects[sha] = write_identical(root, "archive/objects/" + sha + ".json", raw)
        objects[sha]["object_sha256"] = sha
    for asset in assets:
        if "source_logical_path" not in asset:
            continue
        parent = by_name.get(asset["source_logical_path"])
        require(parent is not None and "source_logical_path" not in parent,
                "derived asset must reference a stored canonical WAV")
        data = wav_pcm(object_data[parent["sha256"]])
        require(len(data) == asset["size_bytes"] and digest(data) == asset["sha256"],
                "derived PCM identity mismatch")
    assets.sort(key=lambda a: a["logical_path"])
    asset_pages = page_records(root, "assets", assets)
    object_pages = page_records(root, "object-index", [objects[k] for k in sorted(objects)])
    stats = {"logical_asset_count": len(assets), "logical_bytes": sum(a["size_bytes"] for a in assets),
             "stored_unique_objects": len(objects), "stored_unique_uncompressed_bytes": sum(len(b) for b in object_data.values()),
             "derived_asset_count": sum("source_logical_path" in a for a in assets),
             "stored_unique_shards": len(shards), "stored_shard_bytes": sum(shards.values()),
             "max_shard_bytes": max(shards.values(), default=0)}
    index = {"schema": SCHEMA, "hash_algorithm": "sha256", "shard_limit_bytes": LIMIT,
             "compression": {"format": "gzip", "level": 6, "mtime": 0, "filename": ""},
             "assets": asset_pages, "objects": object_pages, "statistics": stats}
    raw = canonical(index)
    require(len(raw) <= LIMIT, "root index exceeds limit")
    write_identical(root, "archive-index.json", raw)
    return {"archive_index_sha256": digest(raw), **stats}


def load(root, expected_index_sha256=None):
    root = Path(root)
    path = safe_path(root, "archive-index.json")
    require(path.is_file() and path.stat().st_size <= LIMIT, "missing or excessive root index")
    raw = path.read_bytes()
    if expected_index_sha256:
        require(digest(raw) == check_hash(expected_index_sha256), "root index identity mismatch")
    index = json.loads(raw)
    require(index["schema"] == SCHEMA and index["hash_algorithm"] == "sha256"
            and index["shard_limit_bytes"] == LIMIT, "unsupported archive format")
    require(len(index["assets"]) <= MAX_ASSETS and len(index["objects"]) <= MAX_ASSETS,
            "excessive metadata page count")
    assets, object_records = [], []
    for record in index["assets"]:
        assets.extend(json.loads(read_checked(root, record))["assets"])
        require(len(assets) <= MAX_ASSETS, "excessive asset count")
    for record in index["objects"]:
        object_records.extend(json.loads(read_checked(root, record))["object-index"])
        require(len(object_records) <= MAX_ASSETS, "excessive object count")
    by_name, objects = {}, {}
    for asset in assets:
        require("source_path" not in asset, "private source path in public manifest")
        public_entry(asset)
        require(asset["logical_path"] not in by_name, "duplicate logical path")
        by_name[asset["logical_path"]] = asset
    require(sum(a["size_bytes"] for a in assets) <= MAX_TOTAL,
            "archive logical bytes exceed safety limit")
    for name in by_name:
        require(not any(str(p) in by_name for p in PurePosixPath(name).parents if str(p) != "."),
                "logical file/directory collision")
    for record in object_records:
        desc = json.loads(read_checked(root, record))
        sha = check_hash(desc["sha256"])
        require(sha == record["object_sha256"] and sha not in objects, "duplicate/wrong object descriptor")
        check_size(desc["size_bytes"])
        check_size(desc["compressed_size_bytes"], MAX_OBJECT + LIMIT)
        check_hash(desc["compressed_sha256"])
        require(desc["compression"] == "gzip", "unsupported compression")
        require(0 < len(desc["shards"]) <= (MAX_OBJECT // LIMIT + 2), "excessive shard count")
        require(sum(check_size(s["size_bytes"], LIMIT) for s in desc["shards"])
                == desc["compressed_size_bytes"], "compressed length metadata mismatch")
        objects[sha] = desc
    for asset in assets:
        if "source_logical_path" in asset:
            parent = by_name.get(asset["source_logical_path"])
            require(parent is not None and "source_logical_path" not in parent,
                    "missing/derived canonical WAV")
        else:
            require(asset["sha256"] in objects and objects[asset["sha256"]]["size_bytes"] == asset["size_bytes"],
                    "missing object or logical/object size disagreement")
    needed = {a["sha256"] for a in assets if "source_logical_path" not in a}
    require(needed == set(objects), "unreferenced or missing stored objects")
    return index, by_name, objects, digest(raw)


def read_object(root, desc):
    compressed = b"".join(read_checked(root, part) for part in desc["shards"])
    require(digest(compressed) == desc["compressed_sha256"], "compressed object SHA-256 mismatch")
    data = unpack_gzip(compressed, desc["size_bytes"])
    require(digest(data) == desc["sha256"], "object SHA-256 mismatch")
    return data


def verify(root, expected_index_sha256=None, output=None):
    index, assets, objects, index_sha = load(root, expected_index_sha256)
    # Verify every object before writing anything, including objects used only by aliases.
    data = {sha: read_object(root, desc) for sha, desc in objects.items()}
    derived = {}
    for name, asset in assets.items():
        if "source_logical_path" in asset:
            parent = assets[asset["source_logical_path"]]
            pcm = wav_pcm(data[parent["sha256"]])
            require(len(pcm) == asset["size_bytes"] and digest(pcm) == asset["sha256"],
                    "derived PCM identity mismatch")
            derived[name] = pcm
    shards = {s["path"]: s["size_bytes"] for d in objects.values() for s in d["shards"]}
    measured = {"logical_asset_count": len(assets), "logical_bytes": sum(a["size_bytes"] for a in assets.values()),
                "stored_unique_objects": len(objects), "stored_unique_uncompressed_bytes": sum(len(b) for b in data.values()),
                "derived_asset_count": len(derived), "stored_unique_shards": len(shards),
                "stored_shard_bytes": sum(shards.values()), "max_shard_bytes": max(shards.values(), default=0)}
    require(measured == index["statistics"], "statistics mismatch")
    if output is not None:
        output = check_root(output)
        require(not output.exists() or output.is_dir() and not any(output.iterdir()),
                "materialization requires a new or empty directory")
        output.mkdir(parents=True, exist_ok=True)
        for name, asset in sorted(assets.items()):
            write_identical(output, name, derived[name] if name in derived else data[asset["sha256"]])
    return {"status": "PASS", "archive_index_sha256": index_sha, **measured}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("pack", help="pack an explicit local-only approved manifest")
    build.add_argument("plan")
    build.add_argument("root")
    for name in ("verify", "materialize"):
        p = sub.add_parser(name)
        p.add_argument("root")
        p.add_argument("--expected-index-sha256")
        if name == "materialize":
            p.add_argument("output")
    args = parser.parse_args()
    try:
        if args.command == "pack":
            result = pack(args.plan, args.root)
        else:
            result = verify(args.root, args.expected_index_sha256,
                            args.output if args.command == "materialize" else None)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ArchiveError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print("archive error: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
