#!/usr/bin/env python3
"""Offline exact-byte restore; no frontend, model, training, package install or network.

Path checks follow the existing native-a20-packages/tools/restore.py conventions.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import stat
import struct
import subprocess
import tarfile
import zlib

TRANSPORT_SHA256 = '21687bd7d918c6aab02a513234b78f816d11dbeec4111b28052440f281bfc229'
CHUNK_LIMIT = 1048576
ARCHIVE_LIMIT = 4194304
TAR_LIMIT = 8388608


class Stop(ValueError):
    pass


def need(ok, message):
    if not ok:
        raise Stop(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique(pairs):
    out = {}
    for key, value in pairs:
        need(key not in out, 'Duplicate JSON key')
        out[key] = value
    return out


def finite_float(text):
    value = float(text)
    need(math.isfinite(value), 'Nonfinite JSON number')
    return value


def read_json(data):
    return json.loads(data, object_pairs_hook=unique, parse_float=finite_float,
                      parse_constant=lambda _: need(False, 'Nonfinite JSON'))


def path_name(name):
    need(isinstance(name, str) and name and not name.startswith('/') and
         not re.search(r'[\\:\x00-\x1f\x7f<>"|?*]', name), 'Unsafe path')
    parts = name.split('/')
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {
        prefix + n for prefix in ('COM', 'LPT') for n in '123456789'}
    need(all(p not in ('', '.', '..') and p.casefold() != '.git' and
             not p.endswith((' ', '.')) and p.split('.')[0].upper() not in reserved
             for p in parts), 'Unsafe path component')
    return name


def safe_path(root, name=None):
    p = Path(root).absolute()
    if name is not None:
        p = p.joinpath(*path_name(name).split('/'))
    need(not any(q.is_symlink() for q in (p,) + tuple(p.parents)), 'Symlink')
    return p


def identity(data, entry):
    need(type(entry['bytes']) is int and len(data) == entry['bytes'], 'Wrong byte count')
    need(sha(data) == entry['sha256'], 'SHA256 mismatch')


def read_regular(root, entry, bound):
    size = entry['bytes']
    need(type(size) is int and 0 < size <= bound, 'Oversize or invalid file bound')
    p = safe_path(root, entry['path'])
    need(p.is_file() and stat.S_ISREG(p.stat().st_mode), 'Missing regular file')
    need(p.stat().st_size == size, 'Wrong file size')
    with p.open('rb') as f:
        data = f.read(size + 1)
    identity(data, entry)
    return data


def reconstruct(root, archive):
    need(type(archive['bytes']) is int and 0 < archive['bytes'] <= ARCHIVE_LIMIT,
         'Oversize archive')
    if 'chunks' not in archive:
        return read_regular(root, archive, ARCHIVE_LIMIT)
    chunks = archive['chunks']
    need(isinstance(chunks, list) and 1 <= len(chunks) <= 4, 'Chunk count')
    pieces, seen, offset = [], set(), 0
    for ordinal, chunk in enumerate(chunks):
        need(chunk['ordinal'] == ordinal and chunk['offset'] == offset, 'Chunk order/offset')
        need(chunk['path'] == archive['id'] + '/chunks/' + chunk['sha256'] + '.bin',
             'Chunk is not content addressed')
        need(chunk['path'] not in seen, 'Duplicate chunk')
        seen.add(chunk['path'])
        pieces.append(read_regular(root, chunk, CHUNK_LIMIT))
        offset += chunk['bytes']
        need(offset <= archive['bytes'], 'Oversize reconstructed archive')
    directory = safe_path(root, archive['id'] + '/chunks')
    actual = {str(p.relative_to(Path(root))).replace('\\', '/') for p in directory.iterdir()}
    need(actual == seen, 'Extra or missing chunk')
    data = b''.join(pieces)
    identity(data, archive)
    return data


def inspect_archive(data, archive):
    identity(data, archive)
    size = archive['tar_bytes']
    need(type(size) is int and 0 < size <= TAR_LIMIT, 'Oversize tar')
    decoder = zlib.decompressobj(31)
    raw = decoder.decompress(data, size + 1)
    need(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
         'Truncated, concatenated or oversized gzip')
    identity(raw, {'bytes': size, 'sha256': archive['tar_sha256']})
    expected, folded = {}, set()
    for member in archive['members']:
        name = path_name(member['path'])
        need(name not in expected and name.casefold() not in folded, 'Duplicate member')
        need(type(member['bytes']) is int and 0 < member['bytes'] <= CHUNK_LIMIT,
             'Oversize member')
        expected[name] = member
        folded.add(name.casefold())
    need(len(expected) == archive['member_count'] <= 100, 'Member count')
    result, seen, last = {}, set(), 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as tar:
        for member in tar:
            name = path_name(member.name)
            need(name in expected and name not in seen, 'Unexpected/duplicate tar member')
            need(member.isreg() and not member.pax_headers and not member.linkname,
                 'Nonregular or extended tar member')
            need(member.uid == member.gid == member.mtime == 0 and
                 member.uname == member.gname == '', 'Unexpected tar identity metadata')
            need(member.size == expected[name]['bytes'], 'Wrong tar member size')
            payload = tar.extractfile(member).read(member.size + 1)
            identity(payload, expected[name])
            if name.endswith('.json'):
                read_json(payload)
            elif name.endswith('.f32le'):
                need(len(payload) % 4 == 0 and all(math.isfinite(x[0]) for x in
                     struct.iter_unpack('<f', payload)), 'Invalid finite FP32 payload')
            else:
                need(False, 'Unsupported member type')
            result[name] = payload
            seen.add(name)
            last = member.offset_data + ((member.size + 511) // 512) * 512
    need(seen == set(expected), 'Missing tar member')
    need(len(raw) - last >= 1024 and not any(raw[last:]), 'Invalid tar trailer')
    need(sum(map(len, result.values())) == archive['payload_bytes'], 'Payload byte total')
    manifest = read_json(result['artifact-manifest.json'])
    need(manifest['members'] == [expected[n] for n in sorted(expected)
                                 if n != 'artifact-manifest.json'], 'Inner manifest mismatch')
    return result


def verify(root):
    root = safe_path(root)
    manifest_path = root / 'TRANSPORT.json'
    need(manifest_path.is_file() and manifest_path.stat().st_size <= CHUNK_LIMIT, 'Manifest size')
    data = read_regular(root, {'path': 'TRANSPORT.json', 'bytes': manifest_path.stat().st_size,
                             'sha256': TRANSPORT_SHA256}, CHUNK_LIMIT)
    manifest = read_json(data)
    need(manifest['schema'] == 'a20-prepared-inputs-transport-v1' and
         manifest['chunk_limit'] == CHUNK_LIMIT and manifest['archive_limit'] == ARCHIVE_LIMIT and
         manifest['tar_limit'] == TAR_LIMIT, 'Manifest contract')
    need([x['id'] for x in manifest['archives']] == ['success', 'failure'], 'Archive inventory')
    result = {}
    for archive in manifest['archives']:
        blob = reconstruct(root, archive)
        checksum = read_regular(root, archive['checksum'], 200)
        need(checksum == (archive['sha256'] + '  token-preparation-artifact.tar.gz\n').encode(),
             'Original checksum bytes mismatch')
        result[archive['id']] = (blob, inspect_archive(blob, archive))
    return result


def restore(root, output):
    output = safe_path(output)
    need(not output.exists(), 'Output must be a new directory')
    verified = verify(root)  # Validate every byte and path before any output write.
    output.mkdir(parents=True)
    for name, (blob, members) in verified.items():
        base = safe_path(output, name)
        base.mkdir()
        (base / 'token-preparation-artifact.tar.gz').write_bytes(blob)
        for path, payload in members.items():
            target = safe_path(base / 'members', path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f:
                f.write(payload)
    return verified


def check_commit(root, expected):
    need(re.fullmatch(r'[0-9a-f]{40}', expected) is not None, 'Full immutable commit required')
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    need(head == expected, 'Checkout commit mismatch')
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True)
    need(not dirty, 'Consumer checkout must be clean')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'restore'])
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-commit', help='Require clean checkout at this exact data commit')
    args = parser.parse_args()
    if args.expected_commit:
        check_commit(args.root, args.expected_commit)
    need(args.action == 'verify' or args.output is not None, 'Restore needs --output')
    result = verify(args.root) if args.action == 'verify' else restore(args.root, args.output)
    print(json.dumps({'status': 'PASS_SAVED_ARCHIVE_INTEGRITY',
                      'members': {k: len(v[1]) for k, v in result.items()},
                      'model_calls': 0, 'training_updates': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
