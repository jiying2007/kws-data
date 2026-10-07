#!/usr/bin/env python3
"""Retrieve only exact reviewed PyPI inputs through the normal HTTPS route."""
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'asr-runtime-identity-fix/research/asr_runtime'
sys.path.insert(0, str(SOURCE))
import qualify as q

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
os.nice(10)
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 ** 2, 512 * 1024 ** 2))
resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 ** 3, 4 * 1024 ** 3))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

inventory_path = SOURCE / 'locks/inventory.json'
q.require(q.sha256_file(inventory_path) == '626b8c827ee63f95f07e828ae8351ef582197ab253cc68d80b29288679c796e1',
          'reviewed inventory identity changed')
files = q.validate_inventory(q.load_json(inventory_path), 143, 3060297109)
plan_path = SOURCE / 'locks/build-plan.json'
q.require(q.sha256_file(plan_path) == 'fd20f0ddf3e3651880abc05fc851ad244fa839d34066e6b300e6c5c183d19425',
          'reviewed build plan identity changed')
plan = q.load_json(plan_path)
names = set(sys.argv[1:])
if names:
    q.require(names <= {x['name'] for x in files}, 'unknown requested package')
    files = [x for x in files if x['name'] in names]
artifacts, audits = ROOT / 'artifacts', ROOT / 'audits'
artifacts.mkdir(exist_ok=True)
audits.mkdir(exist_ok=True)
opener = urllib.request.build_opener(q.NoRedirect())
deadline = time.monotonic() + 1800
for index, item in enumerate(files, 1):
    available = next(int(x.split()[1]) * 1024 for x in Path('/proc/meminfo').read_text().splitlines()
                     if x.startswith('MemAvailable:'))
    q.require(available >= 3 * 1024 ** 3, 'host memory headroom below 3 GiB')
    q.require(shutil.disk_usage(ROOT).free >= item['bytes'] + 8 * 1024 ** 3,
              'workspace disk reserve below 8 GiB plus next download')
    path = artifacts / item['filename']
    reused = path.exists()
    if reused:
        q.require(path.is_file() and not path.is_symlink() and path.stat().st_size == item['bytes'] and
                  q.sha256_file(path) == item['sha256'], 'existing artifact differs from exact lock')
    else:
        q.download_one(item, path, opener=opener, deadline=deadline)
    audit = q.inspect_archive(path, 12 * 1024 ** 3, 200000, plan['reviewed_pth_sha256'])
    if item['kind'] == 'sdist':
        reviewed = next(x for x in plan['sources'] if x['name'] == item['name'])
        q.require(audit['payload_sha256'] == reviewed['source_payload_manifest_sha256'], 'source payload identity changed')
    else:
        metadata = [v for n, v in audit['payloads'].items() if n.count('/') == 1 and n.endswith('.dist-info/METADATA')]
        q.require(len(metadata) == 1 and metadata[0]['sha256'] == item['metadata_sha256'] and
                  metadata[0]['bytes'] == item['metadata_bytes'], 'wheel metadata differs from reviewed lock')
    receipt = {'name': item['name'], 'version': item['version'], 'filename': item['filename'],
               'bytes': item['bytes'], 'sha256': item['sha256'], 'url': item['url'],
               'expanded_bytes': audit['expanded_bytes'], 'members': audit['members'],
               'payload_sha256': audit['payload_sha256'], 'status': 'hash_and_payload_verified'}
    (audits / (item['filename'] + '.json')).write_bytes(q.canonical(receipt))
    print(json.dumps({'index': index, 'count': len(files), 'reused': reused, **receipt}), flush=True)
(ROOT / 'selected-inputs.json').write_bytes(q.canonical(files))
