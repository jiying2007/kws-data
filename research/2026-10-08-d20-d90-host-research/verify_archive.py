"""Verify retained publication bytes; optionally restore without executing content."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import unicodedata
import zipfile

CHUNK = 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * CHUNK
MAX_OBJECT_BYTES = 128 * CHUNK
MAX_PUBLISHED_BYTES = 1024 * CHUNK
MAX_MEMBERS = 10000
MAX_PARTS = 128
MAX_JSON_BYTES = 16 * CHUNK
SHA256 = re.compile(r'[0-9a-f]{64}\Z')
DEVICE = re.compile(r'(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)', re.I)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_path(text):
    require(type(text) is str and text and len(text) <= 4096, 'Unsafe member path')
    parts = text.split('/')
    require(all(p not in ('', '.', '..') and len(p) <= 255 and
                not p.endswith((' ', '.')) and not DEVICE.match(p) and
                not any(ord(c) < 32 or c in '\\:<>"|?*' for c in p)
                for p in parts), 'Unsafe member path')
    require(unicodedata.normalize('NFC', text) == text, 'Noncanonical Unicode path')
    return PurePosixPath(text)


def checked_inventory(names):
    # Include directory spelling: A/x plus a/y also aliases on Windows/macOS.
    nodes = {}
    files = set()
    for name in names:
        parts = checked_path(name).parts
        for i in range(1, len(parts) + 1):
            path = '/'.join(parts[:i])
            key = path.casefold()
            require(key not in nodes or nodes[key] == path, 'Path alias collision')
            nodes[key] = path
            if i < len(parts):
                require(key not in files, 'File/directory conflict')
        key = name.casefold()
        require(key not in files, 'Duplicate member path')
        require(not any(k.startswith(key + '/') for k in nodes), 'File/directory conflict')
        files.add(key)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def read_json(path):
    with path.open('rb') as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    require(len(data) <= MAX_JSON_BYTES, 'Metadata size limit exceeded')
    return json.loads(data, object_pairs_hook=unique)


def size(value, limit):
    require(type(value) is int and 0 <= value <= limit, 'Invalid or excessive byte count')
    return value


def stream_identity(stream, limit, output=None):
    h = hashlib.sha256()
    count = 0
    while True:
        chunk = stream.read(min(CHUNK, limit - count + 1))
        if not chunk:
            break
        count += len(chunk)
        require(count <= limit, 'Expanded byte limit exceeded')
        h.update(chunk)
        if output is not None:
            output.write(chunk)
    return count, h.hexdigest()


def restore(z, rows, output):
    # POSIX descriptor-relative traversal pins directories and refuses symlinks.
    # Never fall back to path-based writes on unsupported platforms.
    if os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW'):
        raise NotImplementedError('Safe restore requires POSIX directory descriptors')
    output = Path(output)
    parent = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.mkdir(output.name, mode=0o700, dir_fd=parent)
        root_fd = os.open(output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                          dir_fd=parent)
    finally:
        os.close(parent)
    try:
        for row in rows:
            parts = checked_path(row['path']).parts
            fd = os.dup(root_fd)
            try:
                for part in parts[:-1]:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                    child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=fd)
                    os.close(fd)
                    fd = child
                target = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                                 os.O_NOFOLLOW, 0o600, dir_fd=fd)
                with os.fdopen(target, 'wb') as dest, z.open(row['object']) as source:
                    identity = stream_identity(source, row['published_bytes'], dest)
                require(identity == (row['published_bytes'], row['published_sha256']),
                        'Restore identity mismatch')
            finally:
                os.close(fd)
    finally:
        os.close(root_fd)


def verify(root, output=None):
    root = Path(root)
    archive = read_json(root / 'ARCHIVE.json')
    catalog = read_json(root / 'CATALOG.json')
    rows = catalog['members']
    require(type(rows) is list and 0 < len(rows) <= MAX_MEMBERS and
            len(rows) == archive['members'], 'Invalid member inventory')
    checked_inventory([r['path'] for r in rows])
    expected = {}
    total = 0
    for row in rows:
        n = size(row['published_bytes'], MAX_OBJECT_BYTES)
        total += n
        sha = row['published_sha256']
        require(type(sha) is str and SHA256.fullmatch(sha), 'Invalid member digest')
        require(row['object'] == 'objects/' + sha, 'Invalid object identity')
        require(row['object'] not in expected or expected[row['object']] == (n, sha),
                'Conflicting object metadata')
        expected[row['object']] = (n, sha)
        require(type(row['transformations']) is list, 'Invalid transformations')
        require(row['transformations'] or
                (sha == row['original_sha256'] and n == row['original_bytes']),
                'Undeclared member transformation')
    require(total <= MAX_PUBLISHED_BYTES, 'Publication size limit exceeded')
    require(len(expected) == archive['unique_objects'], 'Invalid unique object inventory')
    parts = archive['parts']
    require(type(parts) is list and 0 < len(parts) <= MAX_PARTS, 'Invalid part inventory')
    checked_inventory([p['path'] for p in parts])
    archive_size = size(archive['archive_bytes'], MAX_ARCHIVE_BYTES)
    require(sum(size(p['bytes'], MAX_ARCHIVE_BYTES) for p in parts) == archive_size,
            'Part byte count mismatch')
    # Disk-backed reconstruction bounds RAM; each distinct object is hashed once.
    with tempfile.TemporaryFile() as assembled:
        for part in parts:
            path = root / checked_path(part['path'])
            require(not path.is_symlink() and path.is_file(), 'Part is not a regular file')
            with path.open('rb') as source:
                identity = stream_identity(source, part['bytes'], assembled)
            require(identity == (part['bytes'], part['sha256']),
                    'Part identity mismatch: ' + part['path'])
        assembled.seek(0)
        require(stream_identity(assembled, archive_size) ==
                (archive_size, archive['archive_sha256']), 'Archive identity mismatch')
        assembled.seek(0)
        with zipfile.ZipFile(assembled) as z:
            infos = z.infolist()
            require(len(infos) == len(expected) and {i.filename for i in infos} == set(expected),
                    'Archive object inventory mismatch')
            for info in infos:
                mode = info.external_attr >> 16
                require(not info.is_dir() and stat.S_IFMT(mode) in (0, stat.S_IFREG) and
                        not (info.flag_bits & 1), 'Nonregular or encrypted ZIP object')
                require(info.file_size == expected[info.filename][0], 'Object size mismatch')
                with z.open(info) as source:
                    require(stream_identity(source, info.file_size) == expected[info.filename],
                            'Object identity mismatch')
            if output is not None:
                restore(z, rows, output)
    return {'status': 'PASS_PUBLICATION_BYTES_ONLY', 'members': len(rows),
            'unique_objects': len(expected), 'models_executed': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--restore', type=Path, help='New directory under an existing parent; POSIX only.')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.restore), sort_keys=True))
