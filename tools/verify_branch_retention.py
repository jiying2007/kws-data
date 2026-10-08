#!/usr/bin/env python3
"""Verify retained public Git objects and working bytes; no research execution."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess

MANIFEST = 'docs/BRANCH_RETENTION_20261007.json'
SHA = re.compile(r'[0-9a-f]{40}\Z')
SHA256 = re.compile(r'[0-9a-f]{64}\Z')
# Keep the complete 1,646-file retained proof frozen. New maintenance sources
# are a separate, explicitly reviewed eight-path extension, never a wildcard.
MANIFEST_BYTES = 735260
MANIFEST_BLOB = '8ca411b079f8fc89f53addc1c5c5a7f90219056b'
ATOMIC_SIDECAR = 'research/consolidation/git-atomic-prune-2026-10-08.json'
SIDECAR_MAX_BYTES = 64 * 1024
ATOMIC_NEW_SOURCES = frozenset((
    'tools/git_atomic_prune_20261008.py',
    'tools/git_atomic_guard_20261008.py',
    'tools/git_atomic_askpass_20261008.py',
    'tests/test_git_atomic_prune_20261008.py',
    'tests/test_git_atomic_coordinator_20261008.py',
    'tests/atomic_git_test_fixtures.py',
    'tests/test_git_atomic_api_budget_20261008.py',
))
ATOMIC_EXISTING_SOURCES = frozenset((
    'tools/archive_branches_once_20261007.py',
    'tests/test_archive_branches_once_20261007.py',
    '.github/workflows/archive-branches-once-20261007.yml',
    'tools/verify_branch_retention.py',
    'tests/test_branch_retention.py',
))
ATOMIC_SOURCES = ATOMIC_NEW_SOURCES | ATOMIC_EXISTING_SOURCES
ATOMIC_ADDITIONS = ATOMIC_NEW_SOURCES | {ATOMIC_SIDECAR}


def require(value, message):
    if not value:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def safe_path(value):
    require(type(value) is str and value and '\\' not in value and '\0' not in value,
            'unsafe path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.as_posix() == value and
            not any(part in ('', '.', '..', '.git') for part in path.parts), 'unsafe path')
    return value


def rows_by_path(rows):
    require(type(rows) is list, 'file inventory must be a list')
    result = {}
    for row in rows:
        require(type(row) is dict, 'invalid file row')
        name = safe_path(row['path'])
        require(name not in result, 'duplicate file path')
        require(row['mode'] in ('100644', '100755'), 'unsupported file mode')
        require(type(row['bytes']) is int and row['bytes'] >= 0, 'invalid byte count')
        require(type(row['git_blob_sha1']) is str and SHA.fullmatch(row['git_blob_sha1']),
                'invalid Git blob identity')
        require(type(row['source_commit']) is str and SHA.fullmatch(row['source_commit']),
                'invalid immutable source')
        result[name] = row
    return result


def git_tree(root):
    raw = subprocess.check_output(['git', '-C', str(root), 'ls-tree', '-r', '-z', '--full-tree', 'HEAD'])
    result = {}
    for entry in raw.split(b'\0'):
        if not entry:
            continue
        metadata, name = entry.split(b'\t', 1)
        mode, kind, sha = metadata.decode('ascii').split(' ')
        require(kind == 'blob', 'non-blob entry')
        name = safe_path(name.decode('utf-8'))
        require(name not in result, 'duplicate Git path')
        result[name] = (mode, sha)
    return result


def load_json(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def bound_file(root, name, identity, *, size=None, sha256=None, capture=False, max_bytes=None):
    """Check a regular 100644 working file against its committed Git blob."""
    safe_path(name)
    require(identity[0] == '100644', 'extension Git mode: ' + name)
    require(type(identity[1]) is str and SHA.fullmatch(identity[1]),
            'extension Git blob identity: ' + name)
    path = Path(root)
    require(not path.is_symlink(), 'extension root symlink rejected')
    for part in PurePosixPath(name).parts:
        path = path / part
        require(not path.is_symlink(), 'extension symlink rejected: ' + name)
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode),
            'missing extension regular file: ' + name)
    info = path.stat()
    require(not info.st_mode & 0o111, 'extension working mode: ' + name)
    require(size is None or info.st_size == size, 'extension byte count: ' + name)
    require(max_bytes is None or info.st_size <= max_bytes, 'extension size limit: ' + name)
    # Git's blob digest includes the byte count, binding the sidecar's size
    # without putting a circular self-checksum inside that same sidecar.
    blob = hashlib.sha1(b'blob ' + str(info.st_size).encode('ascii') + b'\0')
    content_hash = hashlib.sha256()
    chunks = [] if capture else None
    read = 0
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            blob.update(chunk)
            content_hash.update(chunk)
            read += len(chunk)
            require(max_bytes is None or read <= max_bytes, 'extension size limit: ' + name)
            if capture:
                chunks.append(chunk)
    require(read == info.st_size and blob.hexdigest() == identity[1],
            'extension committed byte identity: ' + name)
    require(sha256 is None or content_hash.hexdigest() == sha256,
            'extension SHA-256 identity: ' + name)
    return b''.join(chunks) if capture else read


def atomic_extension(root, tree, retained, baseline, integration):
    """Return only the fixed eight additions, all present or all absent."""
    if not ATOMIC_ADDITIONS & set(tree):
        return set()
    require(not ATOMIC_ADDITIONS & (set(retained) | set(baseline) | set(integration)),
            'extension overlaps original inventory')
    require(ATOMIC_ADDITIONS <= set(tree), 'incomplete atomic extension')
    require(ATOMIC_EXISTING_SOURCES <= set(integration),
            'extension source is not original integration')
    raw = bound_file(root, ATOMIC_SIDECAR, tree[ATOMIC_SIDECAR], capture=True,
                     max_bytes=SIDECAR_MAX_BYTES)
    sidecar = load_json(raw)
    require(type(sidecar) is dict and
            sidecar.get('schema') == 'kws-git-atomic-prune-public-source-v1' and
            sidecar.get('repository') == 'jiying2007/kws-data', 'atomic sidecar identity')
    require(type(sidecar.get('files')) is list, 'atomic sidecar file inventory')
    sources = {}
    for row in sidecar['files']:
        require(type(row) is dict and set(row) == {'path', 'mode', 'bytes', 'sha256'},
                'atomic source row fields')
        name = safe_path(row['path'])
        require(name not in sources, 'duplicate atomic source path')
        require(row['mode'] == '100644', 'atomic source mode')
        require(type(row['bytes']) is int and row['bytes'] >= 0, 'atomic source byte count')
        require(type(row['sha256']) is str and SHA256.fullmatch(row['sha256']),
                'atomic source SHA-256')
        sources[name] = row
    require(set(sources) == ATOMIC_SOURCES, 'closed atomic source inventory mismatch')
    require(not set(sources) & (set(retained) | set(baseline)),
            'atomic source overlaps retained or baseline')
    for name, row in sources.items():
        require(name in tree, 'missing atomic source Git file: ' + name)
        bound_file(root, name, tree[name], size=row['bytes'], sha256=row['sha256'])
    return set(ATOMIC_ADDITIONS)


def verify(root, manifest, tree=None):
    root = Path(root)
    require(manifest['schema'] == 'kws-public-branch-retention-v1', 'manifest schema')
    require(SHA.fullmatch(manifest['base_commit']), 'base commit pin')
    rows = rows_by_path(manifest['retained_files'])
    baseline = rows_by_path(manifest['baseline_files'])
    integration = [safe_path(name) for name in manifest['integration_files']]
    require(len(integration) == len(set(integration)), 'duplicate integration path')
    require(not set(integration) & set(rows), 'integration overlaps retained path')
    changes = {safe_path(name) for name in manifest['baseline_changes']}
    require(changes <= set(baseline), 'unknown baseline change')
    for name, old in baseline.items():
        if name not in changes:
            require(name in rows and (rows[name]['mode'], rows[name]['git_blob_sha1']) ==
                    (old['mode'], old['git_blob_sha1']), 'baseline retention mismatch: ' + name)
    require(len(rows) + len(integration) == manifest['expected_tracked_files'], 'tracked count')
    require(len(rows) == manifest['expected_retained_files'], 'retained count')
    tree = git_tree(root) if tree is None else tree
    additions = atomic_extension(root, tree, rows, baseline, integration)
    require(set(tree) == set(rows) | set(integration) | additions,
            'closed tracked-file inventory mismatch')
    verified_bytes = 0
    for name, row in rows.items():
        require(tree[name] == (row['mode'], row['git_blob_sha1']), 'Git identity mismatch: ' + name)
        path = root
        for part in PurePosixPath(name).parts:
            path = path / part
            require(not path.is_symlink(), 'symlink rejected: ' + name)
        require(path.is_file() and stat.S_ISREG(path.stat().st_mode), 'missing regular file: ' + name)
        require(path.stat().st_size == row['bytes'], 'file byte count: ' + name)
        executable = bool(path.stat().st_mode & 0o111)
        require(executable == (row['mode'] == '100755'), 'working mode mismatch: ' + name)
        digest = hashlib.sha1(b'blob ' + str(row['bytes']).encode('ascii') + b'\0')
        read = 0
        with path.open('rb') as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                read += len(chunk)
        require(read == row['bytes'] and digest.hexdigest() == row['git_blob_sha1'],
                'working byte identity mismatch: ' + name)
        verified_bytes += read
    # Integration source bytes are bound by the new commit; reject symlinks here too.
    for name in integration:
        path = root
        for part in PurePosixPath(name).parts:
            path = path / part
            require(not path.is_symlink(), 'integration symlink rejected')
        require(path.is_file(), 'missing integration file')
    return {'status': 'PASS', 'retained_files': len(rows), 'retained_bytes': verified_bytes,
            'tracked_files': len(tree), 'new_model_calls': 0, 'new_network_calls': 0}


def frozen_manifest(root, tree):
    """Load only the unchanged original complete retained-file proof."""
    require(tree.get(MANIFEST) == ('100644', MANIFEST_BLOB), 'frozen retention manifest Git identity')
    raw = bound_file(root, MANIFEST, tree[MANIFEST], size=MANIFEST_BYTES, capture=True)
    return load_json(raw)


def main():
    root = Path(__file__).resolve().parents[1]
    tree = git_tree(root)
    print(json.dumps(verify(root, frozen_manifest(root, tree), tree), sort_keys=True))


if __name__ == '__main__':
    main()
