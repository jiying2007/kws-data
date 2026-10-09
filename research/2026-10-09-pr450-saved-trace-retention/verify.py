"""Verify PR450's saved public bytes only; never execute retained content."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile

ORIGINAL_SHA256 = 'dafaa468c0d9d6ffb323dca2845df74128ea128f0fc3870ce2d4224fcfe26672'
ORIGINAL_BYTES = 23590873
HELPER_SHA256 = 'd6e93e00112916907a5f72717a15286a33e9d50fccd528b41532b34d9afd9605'


def verify(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    helper = root.parent / '2026-10-08-d20-d90-host-research/verify_archive.py'
    if hashlib.sha256(helper.read_bytes()).hexdigest() != HELPER_SHA256:
        raise ValueError('Shared verifier identity drift')
    spec = importlib.util.spec_from_file_location('retained_publication_verifier', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    catalog = module.read_json(root / 'CATALOG.json')
    rows = catalog['members']
    module.require(len(rows) == 1 and rows[0]['path'] == 'frozen-far-source-replay-11085476672.zip'
                   and rows[0]['published_sha256'] == ORIGINAL_SHA256
                   and rows[0]['published_bytes'] == ORIGINAL_BYTES,
                   'PR450 original artifact identity mismatch')
    source = module.read_json(root / 'SOURCE-MEMBERS.json')
    module.require(source['artifact_sha256'] == ORIGINAL_SHA256 and
                   len(source['members']) == 44, 'Original member inventory mismatch')
    members = source['members']
    module.checked_inventory([r['path'] for r in members])
    expected = {r['path']: (r['bytes'], r['sha256']) for r in members}
    with tempfile.TemporaryDirectory() as temp:
        restored = Path(temp) / 'restored'
        module.verify(root, restored)
        archive = restored / rows[0]['path']
        with archive.open('rb') as stream:
            module.preflight_zip(stream, ORIGINAL_BYTES,
                                 {k.encode('ascii'): v for k, v in expected.items()})
        with zipfile.ZipFile(archive) as z:
            module.require(set(z.namelist()) == set(expected), 'Original member names mismatch')
            for info in z.infolist():
                module.require(not info.is_dir() and not (info.flag_bits & 1),
                               'Encrypted or directory source member')
                with z.open(info) as stream:
                    module.require(module.stream_identity(stream, info.file_size) == expected[info.filename],
                                   'Original member byte mismatch: ' + info.filename)
    return {'status': 'PASS_PR450_SAVED_BYTES_ONLY', 'original_members': 44,
            'original_artifact_sha256': ORIGINAL_SHA256, 'models_executed': 0}


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True))
