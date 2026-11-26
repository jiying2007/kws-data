"""Twenty-four whole originals from the two separately registered TRAIN corpora."""
from __future__ import annotations

import hashlib
import io
import json
import math
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
import wave

import english_v6_b28_profile as profile
baseline = profile.preserved('english_v6_b22_natural_sources')
profile.inherit(baseline, globals())
import english_v6_b28_source_ancestry as lineage
import english_v6_b28_source_metadata as metadata
import english_v6_b28_multicorpus_source as multicorpus
qa, digest = lineage.qa, lineage.digest


def plan():
    return lineage.read_plan()[lineage.PANEL]['natural']


def catalogs():
    """Complete original text is external; no selected-only replacement exists."""
    lineage.verify_training()
    directory = profile.proof_root() / 'training-origin/inputs/metadata/train360'
    with metadata.catalog_scope(directory):
        complete = metadata.require_catalog()
        old = metadata._previous().inputs()[0]
        prior = old.read_inputs()['FULL_METADATA_CATALOG.json']
        old.catalog_rows(prior)
        census = (metadata.ROOT / 'automation/english_v6_b21_panels/SPEAKERS.TXT').read_bytes()
        return {key: multicorpus.normalized_catalog(
            prior if key == 'train-clean-100' else complete['catalog'], key, census,
            None if key == 'train-clean-100' else complete['index'])
            for key in sorted({row['archive_key'] for row in plan()['roster']})}


def staging_plan():
    design = plan()
    values = catalogs()
    value = dict(schema='english-v6-b28-multicorpus-staging-plan-v1', panel=lineage.PANEL,
        assessment_sha256=lineage.ASSESSMENT_SHA,
        training_readback_sha256=profile.binding(lineage.PANEL)['training_report_sha256'],
        roster=design['roster'], corpora={key: multicorpus.CORPORA[key] for key in sorted(values)},
        catalog_pins={key: dict(bytes=len(multicorpus.encoded(value)), sha256=multicorpus.sha(multicorpus.encoded(value)))
            for key, value in values.items()}, limits=multicorpus.LIMITS)
    multicorpus.validate_plan(value, values)
    return value, values


def read_staging(directory):
    directory = Path(directory)
    fixed, values = staging_plan()
    qa.need((directory / 'STAGING_PLAN.json').read_bytes() == multicorpus.encoded(fixed), 'exact pretraining frozen current-panel staging plan')
    originals = multicorpus.read_release(directory / 'NATURAL_STAGING/COMPLETE', fixed, values)
    result = profile.document((directory / 'NATURAL_STAGING/RESULT.json').read_bytes())
    qa.need(result['status'] == 'COMPLETE_VERIFIED' and result['panel'] == lineage.PANEL
        and result['schema'] == 'english-v6-b28-multicorpus-staging-result-v1'
        and result['assessment_sha256'] == fixed['assessment_sha256']
        and result['training_readback_sha256'] == fixed['training_readback_sha256']
        and result['plan_sha256'] == multicorpus.sha(multicorpus.encoded(fixed))
        and result['http_attempts'] == sorted(fixed['corpora']) and result['opaque_members_staged'] == 24
        and result['release_sha256'] == qa.sha(directory / 'NATURAL_STAGING/COMPLETE/RELEASE.json')
        and result['compressed_bytes'] == sum(c['bytes'] for c in fixed['corpora'].values())
        and result['selected_bytes_staged'] == sum(i['flac_bytes'] for i in originals)
        and type(result['elapsed_seconds']) in (int, float) and math.isfinite(result['elapsed_seconds'])
        and 0 <= result['elapsed_seconds'] <= multicorpus.LIMITS['maximum_panel_seconds']
        and result['source_budget_authority_granted'] is result['original_compressed_archives_rehashed_offline'] is False,
        'one complete current-panel staging; no source authority inherited from transport')
    started = dict(plan_sha256=multicorpus.sha(multicorpus.encoded(fixed)), panel=lineage.PANEL,
        assessment_sha256=fixed['assessment_sha256'], training_readback_sha256=fixed['training_readback_sha256'],
        maximum_streams=len(fixed['corpora']), maximum_compressed_bytes=sum(c['bytes'] for c in fixed['corpora'].values()),
        status='OPAQUE_STAGING_CONSUMED_UNKNOWN', audio_headers_interpreted=0, native_decodes=0)
    qa.need(profile.document((directory / 'NATURAL_STAGING/STAGING_STARTED.json').read_bytes()) == started,
        'one retained original staging start before any whole stream')
    http = profile.document((directory / 'STAGING_HTTP/RESULT.json').read_bytes())
    qa.need(http == dict(schema='english-v6-b28-whole-original-http-result-v1', panel=lineage.PANEL,
        plan_sha256=multicorpus.sha(multicorpus.encoded(fixed)), status='COMPLETE_VERIFIED',
        attempts=sorted(fixed['corpora']), responses=sorted(fixed['corpora']), retries=0, redirects=0, ranges=0,
        source_budget_authority_granted=False, native_decodes=0, model_calls=0,
        release_sha256=result['release_sha256']), 'every original HTTP attempt and response retained exactly once')
    http_names = {'RESULT.json'}
    for key, corpus in fixed['corpora'].items():
        request = profile.document((directory / 'STAGING_HTTP' / (key + '-GET.json')).read_bytes())
        qa.need(request == dict(method='GET', url=corpus['url'], headers=[['Accept-encoding', 'identity']],
            attempt=1, maximum_gets=1, retries=0, redirects=0, ranges=0), 'one original full-GET request without retry/range')
        response = profile.document((directory / 'STAGING_HTTP' / (key + '-RESPONSE.json')).read_bytes())
        qa.need(set(response) == {'url', 'status', 'headers', 'body_saved', 'redirect_rejected'}
            and response['body_saved'] is response['redirect_rejected'] is False
            and type(response['headers']) is list
            and all(type(pair) is list and len(pair) == 2 and all(type(v) is str for v in pair) for pair in response['headers']),
            'complete original response header pairs')
        headers = Message()
        for name, value in response['headers']:
            headers[name] = value
        multicorpus.validate_response(SimpleNamespace(url=response['url'], status=response['status'], headers=headers), corpus)
        http_names |= {key + '-GET.json', key + '-RESPONSE.json'}
    qa.need(set(profile.inventory(directory / 'STAGING_HTTP')) == http_names,
        'no omitted, retried or undeclared HTTP receipt')
    stage_names = {'STAGING_STARTED.json', 'RESULT.json', 'COMPLETE/RELEASE.json'}
    stage_names |= {'COMPLETE/' + key + '.json' for key in fixed['corpora']}
    stage_names |= {i['path'].relative_to(directory / 'NATURAL_STAGING').as_posix() for i in originals}
    qa.need(set(profile.inventory(directory / 'NATURAL_STAGING')) == stage_names,
        'one complete current-panel staging only, no incomplete or sealed-reserve residue')
    return originals


def extract(source, staging, history):
    source = Path(source)
    design = plan()
    old_pcm, old_groups, old_people = lineage.validate_history(history)
    lineage.require_saved_consumption(source)
    qa.need(Path(staging) == source / 'NATURAL_STAGING' and not set(design['speakers']) & old_people,
        'same current-panel whole originals and fresh readers')
    staged = read_staging(source)
    qa.record(source / 'NATURAL_ROSTERS.json', design['roster'])
    decode_dir = source / 'natural-decodes'
    decode_dir.mkdir()
    originals, seen_pcm = {}, set(old_pcm)
    import soundfile as sf
    for item in staged:
        row = item['row']
        ident, sid = row['source_id'], item['id']
        raw = item['path'].read_bytes()
        qa.need(digest(raw) == item['flac_sha256'] and len(raw) == item['flac_bytes'], 'entire released original FLAC')
        header = flac_header(raw)
        qa.record(decode_dir / (sid + '-STARTED.json'), dict(source_id=ident, source_member=row['flac_member'],
            flac_sha256=digest(raw), attempt=1, status='NATIVE_DECODE_CONSUMED_UNKNOWN'))
        with sf.SoundFile(io.BytesIO(raw)) as audio:
            qa.need((audio.samplerate, audio.channels, audio.subtype, len(audio)) == (16000, 1, 'PCM_16', header['samples']),
                'native whole original matches complete STREAMINFO')
            pcm = audio.read(dtype='int16').astype('<i2').tobytes()
        qa.need(len(pcm) == 2 * header['samples'] and hashlib.md5(pcm).hexdigest() == header['native_pcm_md5'],
            'whole original PCM exactly matches native checksum')
        qa.record(decode_dir / (sid + '-RETURNED.json'), dict(source_id=ident, pcm_sha256=digest(pcm),
            samples=header['samples'], attempt=1, status='COMPLETE_NATIVE_DECODE'))
        qa.need(digest(pcm) not in seen_pcm, 'unseen complete natural PCM')
        seen_pcm.add(digest(pcm))
        with wave.open(str(source / 'wav' / (sid + '.wav')), 'wb') as out:
            out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            out.writeframes(pcm)
        facts = whole_wav_facts(source / 'wav' / (sid + '.wav'))
        group = item['source_group_id']
        qa.need(group not in old_groups, 'independent canonical natural source group')
        originals[ident] = dict(row, **facts, id=sid, source_member=row['flac_member'], flac_sha256=digest(raw),
            retained_flac=item['path'].relative_to(source).as_posix(), native_header=header, source_group_id=group,
            cohort='natural', role='development', requested_text=row['transcript'], requested_class=0,
            observed_class=None, human_gold=False, training_allowed=False, whole_original=True,
            crop_calls=0, pad_calls=0, retry_count=0)
    qa.need(set(originals) == {r['source_id'] for r in design['roster']}, 'all24 frozen originals retained')
    rows, rejected = [], []
    for fixed in design['roster']:
        row = originals[fixed['source_id']]
        if not 8000 <= row['samples'] <= 240000:
            rejected.append(dict(row, reason='duration'))
        elif not row['basic_audio_ok']:
            rejected.append(dict(row, reason='audio_quality'))
        else:
            rows.append(row)
    return dict(rows=rows, source_rejections=rejected, rosters_sha256=qa.sha(source / 'NATURAL_ROSTERS.json'),
        staging_plan_sha256=qa.sha(source / 'STAGING_PLAN.json'),
        staging_release_sha256=qa.sha(source / 'NATURAL_STAGING/COMPLETE/RELEASE.json'),
        all_original_pcm_retained=True, selected_original_slots=24, selected_speakers=12)


def read_source(source, meta, history):
    source, design = Path(source), plan()
    old_pcm, old_groups, old_people = lineage.validate_history(history)
    staged = read_staging(source)
    qa.need(json.loads((source / 'NATURAL_ROSTERS.json').read_bytes()) == design['roster']
        and meta['rosters_sha256'] == qa.sha(source / 'NATURAL_ROSTERS.json')
        and meta['staging_plan_sha256'] == qa.sha(source / 'STAGING_PLAN.json')
        and meta['staging_release_sha256'] == qa.sha(source / 'NATURAL_STAGING/COMPLETE/RELEASE.json')
        and meta['selected_original_slots'] == 24 and meta['selected_speakers'] == 12
        and meta['all_original_pcm_retained'] is True and not set(design['speakers']) & old_people,
        'entire frozen whole-original natural plan and receipts')
    rows, rejected = meta['rows'], meta['source_rejections']
    by_id = {row['source_id']: row for row in rows + rejected}
    qa.need(len(rows) + len(rejected) == len(by_id) == 24 and set(by_id) == {r['source_id'] for r in design['roster']}
        and all(r['reason'] in ('duration', 'audio_quality') for r in rejected), 'every original retained, including every reject')
    rejected_by_id, seen_pcm = {r['source_id']: r for r in rejected}, set(old_pcm)
    for item in staged:
        fixed, row = item['row'], by_id[item['row']['source_id']]
        sid, ident = item['id'], fixed['source_id']
        raw = item['path'].read_bytes()
        header = flac_header(raw)
        path = source / 'wav' / (sid + '.wav')
        facts = whole_wav_facts(path)
        with wave.open(str(path), 'rb') as audio:
            pcm = audio.readframes(audio.getnframes())
        qa.need(all(row[k] == v for k, v in fixed.items()) and all(row[k] == v for k, v in facts.items())
            and row['id'] == sid and row['source_member'] == fixed['flac_member']
            and row['retained_flac'] == item['path'].relative_to(source).as_posix()
            and row['flac_sha256'] == digest(raw) == item['flac_sha256'] and row['native_header'] == header
            and header['samples'] == facts['samples'] and len(pcm) == 2 * facts['samples']
            and hashlib.md5(pcm).hexdigest() == header['native_pcm_md5'], 'all original FLAC/PCM/WAV bytes and full native checksums')
        qa.need(facts['pcm_sha256'] not in seen_pcm and row['source_group_id'] == item['source_group_id']
            and row['source_group_id'] not in old_groups and row['role'] == 'development' and row['cohort'] == 'natural'
            and row['requested_class'] == 0 and row['requested_text'] == fixed['transcript'] and row['observed_class'] is None
            and row['training_allowed'] is row['human_gold'] is False and row['whole_original'] is True
            and row['crop_calls'] == row['pad_calls'] == row['retry_count'] == 0, 'whole fresh natural sources without substitution')
        qa.need(profile.document((source / 'natural-decodes' / (sid + '-STARTED.json')).read_bytes()) ==
            dict(source_id=ident, source_member=fixed['flac_member'], flac_sha256=digest(raw), attempt=1, status='NATIVE_DECODE_CONSUMED_UNKNOWN')
            and profile.document((source / 'natural-decodes' / (sid + '-RETURNED.json')).read_bytes()) ==
            dict(source_id=ident, pcm_sha256=facts['pcm_sha256'], samples=facts['samples'], attempt=1, status='COMPLETE_NATIVE_DECODE'),
            'all24 one-shot native starts and returns')
        seen_pcm.add(facts['pcm_sha256'])
        reason = 'duration' if not 8000 <= facts['samples'] <= 240000 else 'audio_quality' if not facts['basic_audio_ok'] else None
        qa.need((reason is None and ident not in rejected_by_id) or
            (reason is not None and rejected_by_id.get(ident, {}).get('reason') == reason), 'unchanged complete-original rejection')
    qa.need([r['source_id'] for r in rows] == [r['source_id'] for r in design['roster'] if r['source_id'] not in rejected_by_id],
        'only eligible natural originals reach observers, in the frozen order')
    return rows, [by_id[r['source_id']] for r in design['roster']]
