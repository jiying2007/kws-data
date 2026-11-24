"""Bounded HTTP transport measurements; no archive, audio or model is opened."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

REPOSITORY = 'jiying2007/kws-data'
BRANCH = 'refs/heads/research/en-v6-b21-corpus-transport-probe-once-20261124'
PARENT = '60f1e6507216ae605d601f3f09ab15177b46aae2'
TOTAL = 6387309499
LIMIT = 4 * 1024 * 1024
URLS = tuple('https://' + host + '/resources/12/train-clean-100.tar.gz' for host in (
    'www.openslr.org', 'openslr.trmal.net', 'openslr.elda.org', 'openslr.magicdatatech.com'))


def need(value, reason):
    if not value:
        raise ValueError(reason)


def measure(url):
    started = time.monotonic()
    report = dict(url=url, requested_range=f'bytes=0-{LIMIT - 1}', received=0, valid_range=False)
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={'Range': report['requested_range'],
            'Accept-Encoding': 'identity', 'User-Agent': 'kws-pipeline-source-transport-check/1'})
        with urllib.request.urlopen(request, timeout=8) as response:
            report.update(status=response.status, final_url=response.url,
                content_range=response.headers.get('Content-Range'),
                content_length=response.headers.get('Content-Length'),
                etag=response.headers.get('ETag'), last_modified=response.headers.get('Last-Modified'))
            need(response.status == 206 and report['content_range'] == f'bytes 0-{LIMIT - 1}/{TOTAL}'
                 and report['content_length'] == str(LIMIT), 'server must honor the exact bounded range')
            while report['received'] < LIMIT:
                need(time.monotonic() - started < 35, '35-second request wall-clock budget')
                block = response.read(min(64 * 1024, LIMIT - report['received']))
                need(bool(block), 'complete requested byte range')
                digest.update(block)
                report['received'] += len(block)
            need(time.monotonic() - started < 35, '35-second completed request budget')
            report.update(valid_range=True, prefix_sha256=digest.hexdigest())
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error)[:400])
    report['elapsed_seconds'] = time.monotonic() - started
    report['bytes_per_second'] = report['received'] / report['elapsed_seconds']
    return report


def main():
    root = Path(__file__).resolve().parents[1]
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_bytes())
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    parents = subprocess.check_output(['git', '-C', str(root), 'rev-list', '--parents', '-n', '1', 'HEAD'], text=True).split()[1:]
    need(all(os.environ.get(key) == value for key, value in dict(GITHUB_ACTIONS='true',
        GITHUB_REPOSITORY=REPOSITORY, GITHUB_EVENT_NAME='push', GITHUB_REF=BRANCH,
        GITHUB_RUN_ATTEMPT='1', GITHUB_SHA=head).items()) and parents == [PARENT], 'one original public transport-only execution')
    need(event.get('before') == PARENT and event.get('after') == head and event.get('ref') == BRANCH
         and event.get('forced') is event.get('deleted') is False
         and event.get('repository', {}).get('full_name') == REPOSITORY
         and event['repository'].get('private') is False, 'exact authorized public push')
    output = Path(os.environ['RUNNER_TEMP']) / 'b21-transport-probe'
    need(not output.exists(), 'fresh metadata output only')
    output.mkdir()
    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(measure, URLS))
    valid = [item for item in reports if item['valid_range']]
    result = dict(schema='english-v6-b21-bounded-corpus-transport-probe-v1', run_id=os.environ['GITHUB_RUN_ID'],
        run_head=head, prepared_parent=PARENT, whole_archive_bytes=TOTAL, maximum_received_bytes=4 * LIMIT,
        received_bytes=sum(item['received'] for item in reports), reports=reports,
        valid_mirrors=len(valid), valid_prefixes_agree=len({item['prefix_sha256'] for item in valid}) <= 1,
        whole_archive_verified=False, archive_entries_parsed=0, audio_decodes=0, tts_calls=0,
        observer_calls=0, kws_forwards=0, optimizer_updates=0, source_gate_evaluated=False,
        model_promotion_allowed=False)
    (output / 'RESULT.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    need(bool(valid) and result['valid_prefixes_agree'], 'at least one exact-range mirror and no prefix disagreement')


if __name__ == '__main__':
    main()
