"""One fixed whole stream per selected official corpus, after durable staging permit."""
from __future__ import annotations

import signal
from pathlib import Path
import urllib.error
import urllib.request

import english_v6_b28_profile as profile
import english_v6_b28_source_ancestry as lineage
import english_v6_b28_source_metadata as metadata
import english_v6_b28_multicorpus_source as multicorpus


def download(staging, directory, prepared_parent):
    directory, staging = Path(directory), Path(staging)
    lineage.require_staging_permit(directory, prepared_parent)
    profile.need(staging == directory / 'NATURAL_STAGING', 'explicit current-panel staging only')
    fixed, catalogs = lineage.natural_module().staging_plan()
    profile.need((directory / 'STAGING_PLAN.json').read_bytes() == multicorpus.encoded(fixed), 'unchanged staging plan')
    receipts = directory / 'STAGING_HTTP'
    receipts.mkdir()
    result = dict(schema='english-v6-b28-whole-original-http-result-v1', panel=lineage.PANEL,
        plan_sha256=multicorpus.sha(multicorpus.encoded(fixed)), status='INCOMPLETE', attempts=[], responses=[],
        retries=0, redirects=0, ranges=0, source_budget_authority_granted=False, native_decodes=0, model_calls=0)
    collector = metadata._collector()
    opener = urllib.request.build_opener(collector.NoRedirect())
    def response_receipt(key, response):
        lineage.qa.record(receipts / (key + '-RESPONSE.json'), dict(url=response.url, status=response.status,
            headers=list(response.headers.items()), body_saved=False,
            redirect_rejected=bool(getattr(response, 'redirect_rejected', False))))
        result['responses'].append(key)
    def open_response(corpus):
        key = corpus['split']
        profile.need(key in fixed['corpora'] and fixed['corpora'][key] == corpus and key not in result['attempts'],
            'one GET for this exact selected upstream, never a sealed reserve or alternate')
        request = urllib.request.Request(corpus['url'], headers={'Accept-Encoding': 'identity'}, method='GET')
        lineage.qa.record(receipts / (key + '-GET.json'), dict(method='GET', url=request.full_url,
            headers=request.header_items(), attempt=1, maximum_gets=1, retries=0, redirects=0, ranges=0))
        result['attempts'].append(key)
        try:
            response = opener.open(request, timeout=60)
        except urllib.error.HTTPError as error:
            try:
                response_receipt(key, error)
            finally:
                error.close()
            raise
        response_receipt(key, response)
        return response
    def expired(signum, frame):
        raise TimeoutError('fixed 240-minute complete current-panel staging deadline')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, multicorpus.LIMITS['maximum_panel_seconds'])
    try:
        release = multicorpus.stage_panel(fixed, staging, catalogs, open_response)
        profile.need(result['attempts'] == result['responses'] == sorted(fixed['corpora']), 'every and only registered stream once')
        result.update(status='COMPLETE_VERIFIED', release_sha256=multicorpus.sha(multicorpus.encoded(release)))
        return release
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error)[:2048])
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        lineage.qa.record(receipts / 'RESULT.json', result)
