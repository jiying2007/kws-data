#!/usr/bin/env python3
"""Restore the pinned six public A20 ZIPs offline, then verify the original archive.

No Git, credentials, network, inference, training, or benchmark operation exists.
ZIP/path primitives are adapted from the independently reviewed A20 import tool.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import zipfile

TRANSPORT_SHA = '8819c75bacc04182769955700623e3a370ab96984f11282181ab6546d5baa3a7'
PUBLIC_MANIFEST_SHA = 'd0c265888b8e4799101a226eb568f4592f5e63d824493fae4214b0d577b5ed01'
PARTS_SHA = '56375c232ea6408fbe19657ffa632511c11bb3392a1ec40d81d5943a2bef60b1'
ARCHIVE_INDEX_SHA = '3435110711dc158d20ccd3eb1393227090d4a5939e791292e69030a90d78f7e7'
CHUNK_LIMIT = 1048576
PREFIX = 'public/kws-data/'
SHA256 = re.compile(r'[0-9a-f]{64}\Z')


class Stop(Exception):
    pass


def need(ok, message):
    if not ok:
        raise Stop(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def path_name(name):
    need(isinstance(name, str) and name and not name.startswith('/') and
         not re.search(r'[\\:\x00-\x1f\x7f<>"|?*]', name), 'Unsafe member/path name')
    parts = name.split('/')
    need(all(p not in ('', '.', '..') and p.casefold() != '.git' and
             not p.endswith((' ', '.')) for p in parts), 'Unsafe path component')
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {
        prefix + n for prefix in ('COM', 'LPT') for n in '123456789'}
    need(all(p.split('.')[0].upper() not in reserved for p in parts), 'Reserved path name')
    return name


def safe_path(root, name=None):
    p = Path(root).absolute()
    if name is not None:
        p = p.joinpath(*path_name(name).split('/'))
    need(not any(q.is_symlink() for q in (p,) + tuple(p.parents)), 'Symlink in local path')
    return p


def read_regular(path, size=None, digest=None):
    p = safe_path(path)
    need(p.is_file() and stat.S_ISREG(p.stat().st_mode), 'Missing regular file: ' + p.name)
    if size is not None:
        need(p.stat().st_size == size, 'Wrong file size: ' + p.name)
    need(p.stat().st_size <= (size if size is not None else 1024 * 1024),
         'File exceeds expected read bound: ' + p.name)
    data = p.read_bytes()
    if digest:
        need(sha(data) == digest, 'SHA-256 mismatch: ' + p.name)
    return data


def extract_public(parts_dir, output, p):
    """Audit whole ZIP headers; read/extract only exact allowlisted public members."""
    output = safe_path(output)
    need(not output.exists(), 'Extraction requires a new destination')
    seen = set()
    selected = []
    excluded = 0
    total = 0
    # Validate every part and header before creating a single output file.
    for part in p['parts']:
        file = safe_path(parts_dir, part['filename'])
        read_regular(file, part['size_bytes'], part['sha256'])
        with zipfile.ZipFile(file) as z:
            infos = z.infolist()
            need(len(infos) == part['member_count'], 'Wrong ZIP member count: ' + file.name)
            need(sum(i.file_size for i in infos) == part['payload_bytes'], 'Wrong ZIP expansion size')
            names = set()
            folded = set()
            count = 0
            for info in infos:
                name = path_name(info.filename)
                need(name == info.orig_filename, 'Truncated ZIP member name')
                need(name not in names and name.casefold() not in folded, 'Duplicate ZIP member: ' + name)
                names.add(name)
                folded.add(name.casefold())
                mode = info.external_attr >> 16
                need(not info.is_dir() and stat.S_IFMT(mode) in (0, stat.S_IFREG) and
                     not (info.external_attr & 0x10), 'Non-regular ZIP member: ' + name)
                need(not info.flag_bits & 1 and info.compress_type == zipfile.ZIP_STORED,
                     'Encrypted/unsupported ZIP member')
                need(0 <= info.file_size <= 65536 and 0 <= info.compress_size <= part['size_bytes'],
                     'ZIP member exceeds bound')
                need(name.startswith(PREFIX), 'Non-public ZIP member; clean public parts only')
                rel = path_name(name[len(PREFIX):])
                need(rel in p['files'], 'Unapproved public member: ' + rel)
                need(rel not in seen, 'Public member duplicated across parts: ' + rel)
                need(info.file_size == p['files'][rel]['bytes'], 'Wrong public member size: ' + rel)
                seen.add(rel)
                selected.append((file, name, rel))
                count += 1
            need(count == part['member_count'], 'Wrong part public count')
    need(seen == set(p['files']), 'Missing public members')
    output.mkdir(mode=0o700)
    current_file = None
    z = None
    try:
        for file, name, rel in selected:
            if file != current_file:
                if z is not None:
                    z.close()
                z = zipfile.ZipFile(file)
                current_file = file
            rec = p['files'][rel]
            # Reading through EOF causes ZipExtFile to verify the recorded CRC.
            with z.open(name, 'r') as src:
                data = src.read(rec['bytes'] + 1)
                need(len(data) == rec['bytes'] and src.read(1) == b'' and sha(data) == rec['sha256'],
                     'Public CRC/length/SHA failure: ' + rel)
            dest = safe_path(output, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as dst:
                dst.write(data)
            dest.chmod(0o644)
            total += len(data)
    finally:
        if z is not None:
            z.close()
    verify_files(output, p)
    return {'public_files': len(seen), 'public_bytes': total, 'excluded_members': excluded,
            'private_members_extracted': 0, 'parts_verified': len(p['parts'])}


def verify_files(root, p):
    found = set()
    for directory, dirs, files in os.walk(str(root), followlinks=False):
        for name in dirs:
            need(not (Path(directory) / name).is_symlink(), 'Symlink directory in public tree')
        for name in files:
            file = Path(directory) / name
            rel = file.relative_to(root).as_posix()
            need(rel in p['files'], 'Unexpected extracted file: ' + rel)
            rec = p['files'][rel]
            read_regular(file, rec['bytes'], rec['sha256'])
            found.add(rel)
    need(found == set(p['files']), 'Incomplete extracted tree')


def strict_json(data):
    def pairs(items):
        d = {}
        for k, v in items:
            need(k not in d, 'Duplicate JSON key')
            d[k] = v
        return d
    def bad_constant(value):
        raise Stop('Non-finite JSON value')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)


def uint(value, maximum):
    need(type(value) is int and 0 <= value <= maximum, 'Invalid bounded integer')
    return value


def digest(value):
    need(isinstance(value, str) and SHA256.fullmatch(value), 'Invalid SHA-256')
    return value


def validate_chunks(packages, maximum=CHUNK_LIMIT):
    seen = set()
    filenames = set()
    count = 0
    for part in packages:
        name = path_name(part['filename'])
        need('/' not in name and name not in filenames, 'Duplicate or nested ZIP filename')
        filenames.add(name)
        uint(part['size_bytes'], 20 * 1024 * 1024)
        digest(part['sha256'])
        need(isinstance(part['chunks'], list) and 0 < len(part['chunks']) <= 20,
             'Invalid per-package chunk count')
        total = 0
        for ordinal, chunk in enumerate(part['chunks']):
            need(type(chunk['ordinal']) is int and chunk['ordinal'] == ordinal, 'Wrong chunk order')
            size = uint(chunk['size_bytes'], maximum)
            need(size > 0, 'Empty chunk')
            h = digest(chunk['sha256'])
            path = path_name(chunk['path'])
            need(path == 'chunks/' + h + '.bin' and path not in seen, 'Duplicate or non-content-addressed chunk')
            need(ordinal == len(part['chunks']) - 1 or size == maximum, 'Short non-final chunk')
            seen.add(path)
            total += size
            count += 1
        need(total == part['size_bytes'], 'Wrong reconstructed ZIP size')
    return count


def policy(bundle):
    bundle = safe_path(bundle)
    m = strict_json(read_regular(bundle / 'transport.json', digest=TRANSPORT_SHA))
    need(m['schema'] == 'native-a20-public-package-transport-v1' and
         m['chunk_max_bytes'] == CHUNK_LIMIT and m['frozen_archive_index_sha256'] == ARCHIVE_INDEX_SHA,
         'Unexpected transport format')
    for key, filename, expected in [('file_manifest', 'public-files.json', PUBLIC_MANIFEST_SHA),
                                    ('parts_manifest', 'parts.json', PARTS_SHA)]:
        r = m[key]
        need(r['path'] == filename and r['sha256'] == expected, 'Unexpected manifest identity')
        uint(r['size_bytes'], CHUNK_LIMIT)
    fm = strict_json(read_regular(bundle / 'public-files.json', m['file_manifest']['size_bytes'], PUBLIC_MANIFEST_SHA))
    pm = strict_json(read_regular(bundle / 'parts.json', m['parts_manifest']['size_bytes'], PARTS_SHA))
    need(fm['repository'] == pm['repository'] == 'jiying2007/kws-data', 'Wrong repository')
    files, folded = {}, set()
    for rec in fm['files']:
        name = path_name(rec['path'])
        need(name not in files and name.casefold() not in folded, 'Duplicate public path')
        uint(rec['bytes'], 65536)
        digest(rec['sha256'])
        files[name] = rec
        folded.add(name.casefold())
    need(len(files) == fm['file_count'] == m['payload_files'] == 2243 and
         sum(r['bytes'] for r in files.values()) == fm['total_bytes'] == m['payload_bytes'] == 112489844,
         'Wrong frozen physical inventory')
    need(len(m['packages']) == len(pm['parts']) == 6, 'Wrong package count')
    need([{k: v for k, v in r.items() if k != 'chunks'} for r in m['packages']] == pm['parts'],
         'Package identities differ from original clean parts')
    need(validate_chunks(m['packages']) == 111, 'Wrong total chunk count')
    need(sum(r['size_bytes'] for r in m['packages']) == 113247446, 'Wrong total transport bytes')
    return {'files': files, 'parts': m['packages'], 'index_sha': ARCHIVE_INDEX_SHA,
            'research_prefix': path_name(fm['research_prefix'])}


def reconstruct(bundle, output, packages, maximum=CHUNK_LIMIT):
    validate_chunks(packages, maximum)
    bundle, output = safe_path(bundle), safe_path(output)
    chunk_root = safe_path(bundle, 'chunks')
    need(chunk_root.is_dir(), 'Missing chunks directory')
    expected = {c['path'] for part in packages for c in part['chunks']}
    found = set()
    for entry in chunk_root.iterdir():
        need(not entry.is_symlink() and entry.is_file() and stat.S_ISREG(entry.stat().st_mode),
             'Unexpected directory or non-regular chunk entry')
        found.add('chunks/' + entry.name)
    need(found == expected, 'Missing or extra chunk files')
    need(not output.exists(), 'Reconstruction requires a new destination')
    output.mkdir(mode=0o700)
    for part in packages:
        target = safe_path(output, part['filename'])
        total, h = 0, hashlib.sha256()
        with target.open('xb') as dst:
            for c in part['chunks']:
                data = read_regular(safe_path(bundle, c['path']), c['size_bytes'], c['sha256'])
                dst.write(data)
                h.update(data)
                total += len(data)
            dst.flush()
            os.fsync(dst.fileno())
        need(total == part['size_bytes'] and h.hexdigest() == part['sha256'], 'Reconstructed ZIP identity mismatch')


def logical_integrity(physical, logical, p):
    # Physical file hashes have already authenticated the original verifier bytes.
    base = safe_path(physical, p['research_prefix'])
    tool = safe_path(base, 'tools/archive.py')
    report = subprocess.run([sys.executable, '-I', '-B', str(tool), 'materialize', str(base),
                             str(logical), '--expected-index-sha256', p['index_sha']],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    need(report.returncode == 0, 'Original pinned archive materialization failed')
    r = strict_json(report.stdout)
    need(r.get('status') == 'PASS' and r.get('archive_index_sha256') == p['index_sha'] and
         r.get('logical_asset_count') == 341 and r.get('stored_unique_objects') == 301 and
         r.get('derived_asset_count') == 12, 'Unexpected original archive verification result')
    return r


def restore(bundle, output=None):
    p = policy(bundle)
    dest = safe_path(output) if output is not None else None
    if dest is not None:
        need(not dest.exists() and dest.parent.is_dir(), 'Output must be absent under an existing directory')
    with tempfile.TemporaryDirectory(prefix='a20-package-restore-', dir=str(dest.parent) if dest else None) as tmp:
        tmp = Path(tmp)
        reconstruct(bundle, tmp / 'packages', p['parts'])
        # The extracted directory maps directly to the repository root.
        physical = tmp / 'public' / 'kws-data'
        physical.parent.mkdir()
        extraction = extract_public(tmp / 'packages', physical, p)
        logical = logical_integrity(physical, tmp / 'logical-assets', p)
        result = {'status': 'PASS', 'transport_sha256': TRANSPORT_SHA,
                  'public_manifest_sha256': PUBLIC_MANIFEST_SHA, 'chunks_verified': 111,
                  'extraction': extraction, 'logical_archive': logical,
                  'inference_training_benchmark_executed': False}
        if dest is not None:
            # Exclusive creation refuses an output that appeared during verification.
            dest.mkdir(mode=0o700)
            for name in ('packages', 'public', 'logical-assets'):
                os.rename(str(tmp / name), str(safe_path(dest, name)))
            with (dest / 'RESTORE-REPORT.json').open('x', encoding='utf-8') as f:
                json.dump(result, f, indent=2, sort_keys=True)
                f.write('\n')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify', 'restore'))
    parser.add_argument('bundle', nargs='?', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output')
    args = parser.parse_args()
    try:
        need((args.command == 'restore') == (args.output is not None), 'restore requires --output; verify forbids it')
        result = restore(args.bundle, args.output)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (Stop, OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        print('transport error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
