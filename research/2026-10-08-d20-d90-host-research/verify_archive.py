"""Verify retained publication bytes; optionally restore without executing content."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_path(text):
    p = PurePosixPath(text)
    if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in text:
        raise ValueError('Unsafe member path')
    return p


def verify(root, output=None):
    archive = json.loads((root / 'ARCHIVE.json').read_text())
    catalog = json.loads((root / 'CATALOG.json').read_text())
    pieces = []
    for part in archive['parts']:
        data = (root / checked_path(part['path'])).read_bytes()
        if len(data) != part['bytes'] or digest(data) != part['sha256']:
            raise ValueError('Part identity mismatch: ' + part['path'])
        pieces.append(data)
    data = b''.join(pieces)
    if len(data) != archive['archive_bytes'] or digest(data) != archive['archive_sha256']:
        raise ValueError('Archive identity mismatch')
    rows = catalog['members']
    if len(rows) != archive['members'] or len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Invalid member inventory')
    expected = {r['object'] for r in rows}
    if len(expected) != archive['unique_objects']:
        raise ValueError('Invalid unique object inventory')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if len(z.namelist()) != len(expected) or set(z.namelist()) != expected:
            raise ValueError('Archive object inventory mismatch')
        for name in sorted(expected):
            checked_path(name)
            if digest(z.read(name)) != name.removeprefix('objects/'):
                raise ValueError('Object identity mismatch')
        for row in rows:
            checked_path(row['path'])
            payload = z.read(row['object'])
            if len(payload) != row['published_bytes'] or digest(payload) != row['published_sha256']:
                raise ValueError('Member identity mismatch: ' + row['path'])
            if not row['transformations'] and (row['published_sha256'] != row['original_sha256'] or row['published_bytes'] != row['original_bytes']):
                raise ValueError('Undeclared member transformation')
        if output:
            output.mkdir(parents=True, exist_ok=False)
            for row in rows:
                dest = output / checked_path(row['path'])
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(row['object']))
    return {'status': 'PASS_PUBLICATION_BYTES_ONLY', 'members': len(rows), 'unique_objects': len(expected), 'models_executed': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--restore', type=Path, help='New destination directory; refuses an existing directory.')
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.restore), sort_keys=True))
