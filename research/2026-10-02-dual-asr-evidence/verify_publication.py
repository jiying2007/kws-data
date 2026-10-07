#!/usr/bin/env python3
"""Verify this public projection without importing model code or using network."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
inventory=ROOT/'SHA256SUMS'
expected={}
for line in inventory.read_text().splitlines():
    digest,path=line.split('  ',1)
    if len(digest)!=64 or Path(path).is_absolute() or '..' in Path(path).parts:
        raise ValueError('Invalid checksum entry')
    if path in expected: raise ValueError('Duplicate checksum entry')
    expected[path]=digest
actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
# Historical child checksums themselves are covered by this inventory.
actual |= {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('SHA256SUMS') if p!=inventory}
if actual!=set(expected): raise ValueError({'missing':sorted(set(expected)-actual),'extra':sorted(actual-set(expected))})
for path,digest in expected.items():
    p=ROOT/path
    if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
        raise ValueError('File digest mismatch: '+path)
inputs=json.loads((ROOT/'input-references.json').read_text())
repo=ROOT.parents[1]
for item in inputs:
    p=repo/item['path']
    data=p.read_bytes()
    if p.is_symlink() or len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
        raise ValueError('Preserved WAV mismatch: '+item['clip_id'])
print(json.dumps({'status':'pass','published_files':len(expected),'preserved_input_wavs':len(inputs),'model_forwards':0}))
