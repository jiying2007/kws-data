"""Reuse a complete original proof only inside one independent audit scope.

This is a byte-identity optimization, not an admission or permission decision.
Every call reopens and hashes the entire exact evidence closure and reader.
The first call must run its original verifier and reproduce the pinned canonical
report; its inputs are checked again before immutable report bytes are kept.
No value survives the outer scope, and an unscoped call always executes.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

KINDS = frozenset(('r1_source_stop', 'q1_primary_quality'))
_ACTIVE = ContextVar('b24q1_independent_evidence_audit', default=None)
_CHUNK = 1024 * 1024


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hash_string(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('ascii')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _need(key not in result, 'duplicate JSON key in scoped evidence')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('nonfinite JSON constant in scoped evidence: ' + value)


def _float(value):
    number = float(value)
    _need(math.isfinite(number), 'overflowed JSON number in scoped evidence')
    return number


def _document(raw, role):
    _need(type(raw) is bytes and len(raw) > 0, 'complete immutable ' + role + ' bytes')
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs,
                       parse_constant=_constant, parse_float=_float)
    _need(type(value) is dict and raw == _canonical(value),
          'exact canonical JSON object for ' + role)
    return value


def _absolute(path):
    path = Path(path)
    _need('..' not in path.parts, 'no parent traversal in scoped evidence path')
    return Path(os.path.abspath(path))


def _regular_ancestors(path, final_directory=False):
    """Check lexical ancestors; resolving first would hide a symlink."""
    path = _absolute(path)
    for parent in reversed(path.parents):
        mode = parent.lstat().st_mode
        _need(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode),
              'no symlink or non-directory evidence ancestor: ' + str(parent))
    mode = path.lstat().st_mode
    expected = stat.S_ISDIR(mode) if final_directory else stat.S_ISREG(mode)
    _need(expected and not stat.S_ISLNK(mode),
          'regular evidence ' + ('directory: ' if final_directory else 'file: ') + str(path))
    return path


def _pins(files):
    _need(type(files) is dict and len(files) > 0, 'complete nonempty evidence file map')
    result = {}
    directories = set()
    for name, pin in files.items():
        _need(type(name) is str and name and '\\' not in name and '\x00' not in name,
              'safe relative evidence path')
        path = PurePosixPath(name)
        _need(not path.is_absolute() and path.as_posix() == name
              and all(part not in ('', '.', '..') for part in name.split('/')),
              'normalized relative evidence path')
        _need(type(pin) is dict and set(pin) == {'bytes', 'sha256'}
              and type(pin['bytes']) is int and pin['bytes'] >= 0
              and _hash_string(pin['sha256']), 'exact evidence byte pin: ' + name)
        result[name] = {'bytes': pin['bytes'], 'sha256': pin['sha256']}
        directories.update(p.as_posix() for p in path.parents if p.as_posix() != '.')
    _need(not set(result) & directories, 'evidence file cannot also be a directory')
    return result, directories


def _closure(directory, files, directories):
    directory = _regular_ancestors(directory, final_directory=True)
    found_files, found_directories = set(), set()
    for current, children, names in os.walk(directory, followlinks=False):
        current = Path(current)
        for name in children:
            path = current / name
            mode = path.lstat().st_mode
            _need(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode),
                  'no symlink or special evidence directory: ' + str(path))
            found_directories.add(path.relative_to(directory).as_posix())
        for name in names:
            path = current / name
            mode = path.lstat().st_mode
            _need(stat.S_ISREG(mode) and not stat.S_ISLNK(mode),
                  'no symlink or special evidence member: ' + str(path))
            found_files.add(path.relative_to(directory).as_posix())
    _need(found_files == set(files) and found_directories == directories,
          'exact complete evidence directory closure')
    return directory


def _file_hash(path, expected_size=None):
    path = _regular_ancestors(path)
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        _need(stat.S_ISREG(before.st_mode), 'opened evidence file is regular')
        if expected_size is not None:
            _need(before.st_size == expected_size, 'whole evidence file byte count: ' + str(path))
        digest, size = hashlib.sha256(), 0
        while True:
            raw = os.read(descriptor, _CHUNK)
            if not raw:
                break
            size += len(raw)
            _need(expected_size is None or size <= expected_size,
                  'evidence file grew while hashing: ' + str(path))
            digest.update(raw)
        after = os.fstat(descriptor)
        current = path.lstat()
        # These fields only detect changes while reading. They never replace
        # a whole-file hash and are never a cache identity or cache shortcut.
        fields = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        _need(all(getattr(before, field) == getattr(after, field) == getattr(current, field)
                  for field in fields), 'evidence file changed while hashing: ' + str(path))
        _need(size == before.st_size, 'read every complete evidence byte: ' + str(path))
        return {'bytes': size, 'sha256': digest.hexdigest()}
    finally:
        os.close(descriptor)


def _authenticate(directory, files, directories, reader_path, reader_sha256):
    directory = _closure(directory, files, directories)
    for name, pin in sorted(files.items()):
        _need(_file_hash(directory / name, pin['bytes']) == pin,
              'whole evidence file SHA-256: ' + name)
    _closure(directory, files, directories)
    _need(_file_hash(reader_path)['sha256'] == reader_sha256,
          'unchanged complete original reader SHA-256')


@contextmanager
def scope():
    """Nested calls share this audit; the outer exit destroys every result."""
    if _ACTIVE.get() is not None:
        yield
        return
    cache = {}
    token = _ACTIVE.set(cache)
    try:
        yield
    finally:
        cache.clear()
        _ACTIVE.reset(token)


def verify_once(kind, directory, files, lock_raw, reader_path,
                reader_sha256, report_sha256, execute):
    """Return only an exact original report after full byte authentication.

    Caller-specific lock semantics, role caps, PASS/failure meaning and
    authority remain the responsibility of the unchanged admission code.
    Changing any lock or pin forces a new full original verification.
    """
    _need(type(kind) is str and kind in KINDS, 'known original evidence kind')
    _document(lock_raw, 'original lock')
    files, directories = _pins(files)
    _need(_hash_string(reader_sha256) and _hash_string(report_sha256),
          'exact original reader and report SHA-256 pins')
    _need(callable(execute), 'original complete verification callback')
    # Snapshot caller-owned mutable inputs before executing its verifier.
    directory, reader_path = _absolute(directory), _absolute(reader_path)
    key = (kind, lock_raw, _canonical(files), reader_sha256, report_sha256)
    cache = _ACTIVE.get()
    try:
        _authenticate(directory, files, directories, reader_path, reader_sha256)
        if cache is not None and key in cache:
            return cache[key]
        raw = execute()
        _document(raw, 'original report')
        _need(_sha(raw) == report_sha256, 'exact original canonical report SHA-256')
        _authenticate(directory, files, directories, reader_path, reader_sha256)
        if cache is not None:
            cache[key] = raw
        return raw
    except BaseException:
        if cache is not None:
            cache.pop(key, None)
        raise
