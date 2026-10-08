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
    require(set(tree) == set(rows) | set(integration), 'closed tracked-file inventory mismatch')
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


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / MANIFEST).read_bytes(), object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    print(json.dumps(verify(root, manifest), sort_keys=True))


if __name__ == '__main__':
    main()
