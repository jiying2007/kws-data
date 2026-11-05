"""Persistent private speaker/split assignments; HMAC is migration-only.

Run once before source intake, and extend the same registry for new speakers.
Existing assignments are immutable. A missing registry is never silently
replaced with newly randomized identities.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import uuid

from .identity import ROLES, SourceRoles

SCHEMA = "kws-private-speaker-registry-v1"
POLICY = "append-three-train-one-development-v1"
IDENT = re.compile(r"[a-z][a-z0-9_-]{1,63}\Z")
MAX_BYTES = 8 * 1024 * 1024


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _pairs(rows):
    out = {}
    for key, value in rows:
        require(key not in out, "REGISTRY_DUPLICATE_JSON_KEY")
        out[key] = value
    return out


def read_bytes(path, limit=MAX_BYTES):
    require(type(limit) is int and limit > 0, "REGISTRY_READ_LIMIT")
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)),
            "REGISTRY_PATH_ALIAS")
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
                0 < info.st_size <= limit, "REGISTRY_FILE")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    require(len(raw) == info.st_size and
            all(getattr(info, key) == getattr(after, key) for key in fields),
            "REGISTRY_CHANGED_DURING_READ")
    return raw


def read_json(path):
    return json.loads(read_bytes(path), object_pairs_hook=_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(
                          ValueError("REGISTRY_NONFINITE_JSON")))


def _speaker_key(dataset_id, speaker_key):
    require(type(dataset_id) is str and IDENT.fullmatch(dataset_id),
            "REGISTRY_DATASET_ID")
    require(type(speaker_key) is str and 1 <= len(speaker_key) <= 128 and
            speaker_key == speaker_key.strip() and
            all(ord(c) >= 32 for c in speaker_key), "REGISTRY_SPEAKER_KEY")
    return dataset_id, speaker_key


def validate(registry):
    require(type(registry) is dict and set(registry) == {
        "schema", "registry_id", "split_policy", "entries"} and
        registry["schema"] == SCHEMA and registry["split_policy"] == POLICY and
        type(registry["registry_id"]) is str and
        IDENT.fullmatch(registry["registry_id"]), "SPEAKER_REGISTRY_SCHEMA")
    rows = registry["entries"]
    require(type(rows) is list and len(rows) <= 10000, "REGISTRY_SIZE")
    by_key, families, speakers = {}, set(), set()
    roles = SourceRoles()
    for row in rows:
        require(type(row) is dict and set(row) == {
            "dataset_id", "speaker_key", "speaker_group_id", "source_family_id",
            "role"}, "REGISTRY_ENTRY")
        key = _speaker_key(row["dataset_id"], row["speaker_key"])
        require(key not in by_key, "REGISTRY_DUPLICATE_SPEAKER")
        for field in ("speaker_group_id", "source_family_id"):
            require(type(row[field]) is str and IDENT.fullmatch(row[field]),
                    "REGISTRY_SOURCE_ID")
        require(row["speaker_group_id"] not in speakers and
                row["source_family_id"] not in families, "REGISTRY_ID_ALIAS")
        roles.add(row["role"], family=row["source_family_id"],
                  speaker=row["speaker_group_id"])
        speakers.add(row["speaker_group_id"])
        families.add(row["source_family_id"])
        by_key[key] = row
    return by_key


def new_registry():
    return {"schema": SCHEMA, "registry_id": "registry-" + uuid.uuid4().hex,
            "split_policy": POLICY, "entries": []}


def extend(registry, dataset_id, speaker_keys):
    """Append in canonical order; never recompute an existing speaker's role."""
    by_key = validate(registry)
    keys = sorted({_speaker_key(dataset_id, key) for key in speaker_keys})
    result = copy.deepcopy(registry)
    for dataset, speaker in keys:
        if (dataset, speaker) in by_key:
            continue
        token = uuid.uuid4().hex
        result["entries"].append({
            "dataset_id": dataset, "speaker_key": speaker,
            "speaker_group_id": "spk-" + token, "source_family_id": "fam-" + token,
            "role": ("development-only" if len(result["entries"]) % 4 == 3
                     else "train-candidate")})
    validate(result)
    return result


def lookup(registry, dataset_id, speaker_key):
    by_key = validate(registry)
    key = _speaker_key(dataset_id, speaker_key)
    require(key in by_key, "SPEAKER_NOT_REGISTERED")
    return dict(by_key[key])


def migrate_legacy(index, manifest, legacy_key):
    """One-time adoption of old IDs/roles, without changing historical files.

    The old private key is used only to join original speaker keys to the old
    manifest. It is never stored or used for any new split assignment.
    """
    require(type(legacy_key) is bytes and 32 <= len(legacy_key) <= 128,
            "LEGACY_KEY_REQUIRED_FOR_IDENTITY_JOIN")
    require(type(manifest) is dict and
            manifest.get("schema") == "kws-bulk-clone-sources-v1" and
            type(manifest.get("references")) is list, "LEGACY_SOURCE_MANIFEST")
    dataset = index["dataset_id"]
    originals = {}
    for row in index["rows"]:
        key = _speaker_key(dataset, row["speaker_key"])
        originals.setdefault(key, set()).add(row["original_sha256"])
    result = new_registry()
    seen = set()
    for (dataset, speaker), hashes in sorted(originals.items()):
        token = hmac.new(legacy_key, (dataset + "\0" + speaker).encode("utf-8"),
                         hashlib.sha256).hexdigest()[:24]
        refs = [row for row in manifest["references"]
                if row.get("dataset_id") == dataset and
                row.get("speaker_group_id") == "spk-" + token and
                row.get("source_family_id") == "fam-" + token]
        if not refs:
            continue  # The original index may contain unselected speakers.
        require(all(row.get("parent_source_sha256") in hashes for row in refs),
                "LEGACY_ORIGINAL_SPEAKER_MISMATCH")
        require(len({row.get("role") for row in refs}) == 1 and
                refs[0].get("role") in ROLES, "LEGACY_SPLIT_CONFLICT")
        seen.update(row["reference_id"] for row in refs)
        result["entries"].append({
            "dataset_id": dataset, "speaker_key": speaker,
            "speaker_group_id": refs[0]["speaker_group_id"],
            "source_family_id": refs[0]["source_family_id"],
            "role": refs[0]["role"]})
    require(seen and seen == {row["reference_id"] for row in manifest["references"]},
            "LEGACY_REFERENCE_JOIN_INCOMPLETE")
    validate(result)
    return result


def save(path, proposed, previous):
    """Atomic, compare-and-swap update; existing assignments are append-only."""
    validate(proposed)
    path = Path(path).absolute()
    require(path.parent.is_dir() and
            not any(p.is_symlink() for p in (path, *path.parents)),
            "REGISTRY_PATH_ALIAS")
    if previous is not None:
        validate(previous)
        require(proposed["registry_id"] == previous["registry_id"] and
                proposed["entries"][:len(previous["entries"])] == previous["entries"],
                "EXISTING_SOURCE_ASSIGNMENT_CHANGED")
    lock_path = path.with_name(path.name + ".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "rb") as lock:
        require(stat.S_ISREG(os.fstat(lock.fileno()).st_mode) and
                os.fstat(lock.fileno()).st_nlink == 1, "REGISTRY_LOCK_ALIAS")
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = read_json(path) if path.exists() else None
        require(current == previous, "REGISTRY_CHANGED_CONCURRENTLY")
        name = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".registry-",
                                             delete=False) as temp:
                name = temp.name
                temp.write(canonical(proposed) + b"\n")
                temp.flush()
                os.fsync(temp.fileno())
            os.replace(name, path)
            name = None
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if name is not None:
                os.unlink(name)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--registry", required=True, type=Path)
    cli.add_argument("--index", required=True, type=Path)
    cli.add_argument("--initialize", action="store_true")
    cli.add_argument("--legacy-manifest", type=Path)
    cli.add_argument("--legacy-key-file", type=Path)
    cli.add_argument("--execute", action="store_true")
    args = cli.parse_args(argv)
    index = read_json(args.index)
    existing = read_json(args.registry) if args.registry.exists() else None
    require((existing is None) == args.initialize,
            "INITIALIZE_ONLY_NEW_REGISTRY_OTHERWISE_EXTEND_EXISTING")
    require(bool(args.legacy_manifest) == bool(args.legacy_key_file),
            "LEGACY_JOIN_REQUIRES_MANIFEST_AND_KEY")
    if args.legacy_manifest:
        require(existing is None, "LEGACY_MIGRATION_REQUIRES_NEW_REGISTRY")
        key_path = args.legacy_key_file
        require(not any(p.is_symlink() for p in (key_path, *key_path.parents)),
                "LEGACY_KEY_ALIAS")
        require(key_path.is_file() and key_path.stat().st_nlink == 1 and
                32 <= key_path.stat().st_size <= 128, "LEGACY_KEY_FILE")
        proposed = migrate_legacy(index, read_json(args.legacy_manifest),
                                  key_path.read_bytes())
    else:
        proposed = extend(existing if existing is not None else new_registry(),
                          index["dataset_id"], [r["speaker_key"] for r in index["rows"]])
    if args.execute:
        root = Path(__file__).resolve().parents[1]
        target = args.registry.absolute()
        require(target != root and root not in target.parents, "REGISTRY_PRIVATE_ONLY")
        save(target, proposed, existing)
    print(json.dumps({"status": "SAVED" if args.execute else "PREVIEW",
                      "speakers": len(proposed["entries"]),
                      "new_speakers": len(proposed["entries"]) -
                          (len(existing["entries"]) if existing else 0)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
