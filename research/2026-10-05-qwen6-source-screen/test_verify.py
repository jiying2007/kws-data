"""Bounded offline rejection checks for the prepared saved archive."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT


def rebind(root, relative):
    path = root / 'archive-manifest.json'
    manifest = json.loads(path.read_text())
    data = (root / relative).read_bytes()
    row = next(x for x in manifest['files'] if x['path'] == relative)
    row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
               git_blob_sha1=hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest())
    path.write_text(json.dumps(manifest, indent=2) + '\n')


results = []
for case in ('raw_chunk_byte', 'rebound_result_claim', 'rebound_gate_claim', 'existing_restore'):
    with tempfile.TemporaryDirectory(prefix='qwen6-reject-') as tmp:
        root = Path(tmp) / 'archive'
        shutil.copytree(SOURCE, root)
        args = ['python', '-B', str(root / 'verify.py')]
        if case == 'raw_chunk_byte':
            p = root / 'raw/tts-native-recovery-v3.zip.part000'
            data = bytearray(p.read_bytes()); data[-1] ^= 1; p.write_bytes(data)
        elif case == 'rebound_result_claim':
            name = 'result-projection.json'; p = root / name
            value = json.loads(p.read_text()); value['rows'][0]['human_truth_established'] = True
            p.write_text(json.dumps(value)); rebind(root, name)
        elif case == 'rebound_gate_claim':
            name = 'panel-integration/result/GATE-REPORT.json'; p = root / name
            value = json.loads(p.read_text()); value['status'] = 'PASS'
            p.write_text(json.dumps(value)); rebind(root, name)
        else:
            destination = Path(tmp) / 'existing'; destination.mkdir()
            marker = destination / 'keep.txt'; marker.write_text('unchanged')
            args += ['--restore', str(destination)]
        outcome = subprocess.run(args, capture_output=True, text=True)
        assert outcome.returncode != 0, case
        if case == 'existing_restore':
            assert marker.read_text() == 'unchanged'
        results.append({'case': case, 'status': 'REJECTED_AS_EXPECTED'})
print(json.dumps({'status': 'PASS', 'cases': results, 'model_calls': 0, 'network_calls': 0}))
