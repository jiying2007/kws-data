"""Same 256 MB SOURCE limit, measured whole originals before native consumption."""
from __future__ import annotations

import importlib
from pathlib import Path
import english_v6_b28_profile as profile

base = profile.preserved('english_v6_b24q1_capacity')
old_capacity = profile.preserved('english_v6_b27_capacity')
need, sha, canonical = profile.need, profile.sha, profile.canonical
SOURCE_LIMIT, CORE_LIMIT = profile.LIMITS['source'], profile.LIMITS['core']
SCHEMA = 'english-v6-b28-prospective-source-capacity-v1'


def natural_module(panel):
    need(panel in ('primary', 'replication'), 'explicit fixed panel')
    return importlib.import_module('english_v6_b28_natural_sources' if panel == 'primary'
        else 'english_v6_b28_replication_natural_sources')


def natural_envelope(directory, panel):
    natural = natural_module(panel)
    result = {}
    for item in natural.read_staging(directory):
        with item['path'].open('rb') as source:
            raw = source.read(42)
        need(len(raw) == 42 and raw[:4] == b'fLaC' and raw[4] & 0x7f == 0
             and int.from_bytes(raw[5:8], 'big') == 34, 'complete original FLAC STREAMINFO before decode')
        packed = int.from_bytes(raw[18:26], 'big')
        samples = packed & ((1 << 36) - 1)
        need(packed >> 44 == 16000 and ((packed >> 41) & 7) + 1 == 1
             and ((packed >> 36) & 31) + 1 == 16 and samples > 0 and raw[26:42] != bytes(16), 'whole mono16k PCM16 identity')
        name = item['path'].relative_to(directory).as_posix()
        result[name] = dict(source_id=item['row']['source_id'], flac_bytes=item['flac_bytes'], samples=samples,
            wav_bytes=2 * samples + 44, streaminfo_sha256=sha(raw))
    need(len(result) == 24, 'all24 whole native originals measured, without a decode')
    return dict(members=result, bytes=sum(v['flac_bytes'] + v['wav_bytes'] for v in result.values()),
        original_count=24, duration_rejects_retained=True)


def budget(directory, staging, natural_plan, panel):
    directory, staging = Path(directory), Path(staging)
    profile.require_proof(panel)
    need(staging == directory / 'NATURAL_STAGING', 'exact whole-original staging under authorization root')
    measured = natural_envelope(directory, panel)
    actual = base.inventory(directory)['files']
    known = {n: p for n, p in actual.items() if n not in ('CAPACITY.json', 'CONSUMPTION.json', 'CONSUMPTION_UPLOAD.json')
        and n not in measured['members']}
    need({'AUTHORIZATION.json', 'HISTORY.json', 'PROOF_BINDING.json', 'STAGING_PLAN.json',
        'STAGING_CONSUMPTION.json', 'STAGING_CONSUMPTION_UPLOAD.json', 'NATURAL_STAGING/COMPLETE/RELEASE.json'} <= set(known)
        and not any(n.endswith('.zip') or n.startswith(('wav/', 'native/')) for n in known),
        'external originals are bound once; full current staged originals are separately accounted')
    expanded = sum(p['bytes'] for p in known.values()) + base.TTS_AUDIO_BOUND + measured['bytes'] + base.NEW_METADATA_LIMIT
    packed = base.zip_bound(expanded, base.MAX_MEMBERS, base.MAX_MEMBERS * base.MAX_PATH_BYTES)
    need(expanded <= SOURCE_LIMIT and packed <= SOURCE_LIMIT, 'STOP before native consumption: whole SOURCE exceeds unchanged256MB')
    return dict(schema=SCHEMA, panel=panel, limit_bytes=SOURCE_LIMIT, original_artifact_limit_bytes=CORE_LIMIT,
        external_originals_retained=True, proof_binding_sha256=sha((directory / 'PROOF_BINDING.json').read_bytes()),
        known_files=known, natural=measured, natural_plan_sha256=sha(canonical(natural_plan)),
        maximum_members=base.MAX_MEMBERS, maximum_path_bytes=base.MAX_PATH_BYTES,
        tts_audio_bytes_max=base.TTS_AUDIO_BOUND, new_metadata_bytes_max=base.NEW_METADATA_LIMIT,
        maximum_expanded_bytes=expanded, maximum_zip_bytes=packed, no_member_truncation=True,
        no_scientific_gate_changed=True, new_model_calls=0, new_source_calls=0, new_observer_calls=0)


def before_consumption(directory, staging, natural_plan, panel):
    need(not (Path(directory) / 'CAPACITY.json').exists() and not (Path(directory) / 'CONSUMPTION.json').exists(),
        'one complete whole-original capacity before native budget')
    return budget(directory, staging, natural_plan, panel)


def before_generation(directory, staging, natural_plan, panel):
    expected = budget(directory, staging, natural_plan, panel)
    need(profile.document((Path(directory) / 'CAPACITY.json').read_bytes()) == expected, 'whole originals unchanged after capacity freeze')
    return expected


def source_members(directory, panel):
    directory = Path(directory)
    value = profile.document((directory / 'CAPACITY.json').read_bytes())
    need(value['schema'] == SCHEMA and value['panel'] == panel and value['limit_bytes'] == SOURCE_LIMIT
        and value['original_artifact_limit_bytes'] == CORE_LIMIT and value['tts_audio_bytes_max'] == base.TTS_AUDIO_BOUND
        and value['new_metadata_bytes_max'] == base.NEW_METADATA_LIMIT, 'same prospective source byte ceiling')
    current = base.inventory(directory)
    files = current['files']
    need(all(files.get(n) == p for n, p in value['known_files'].items())
        and value['proof_binding_sha256'] == sha((directory / 'PROOF_BINDING.json').read_bytes())
        and not any(n.endswith('.zip') for n in files), 'all original source metadata retained without historical archive nesting')
    measured = natural_envelope(directory, panel)
    need(measured == value['natural'], 'every original retained FLAC and exact native sample count unchanged')
    natural_wavs = set()
    for name, row in measured['members'].items():
        wav = 'wav/b28-' + panel + '-natural-' + row['source_id'] + '.wav'
        need(files.get(name, {}).get('bytes') == row['flac_bytes'] and files.get(wav, {}).get('bytes') == row['wav_bytes'],
            'whole natural FLAC and PCM bytes, including rejected originals')
        natural_wavs.add(wav)
    wav_names = {n for n in files if n.startswith('wav/')}
    synthetic = wav_names - natural_wavs
    need(len(wav_names) == 120 and len(synthetic) == 96
        and all(files[n]['bytes'] <= 2 * base.TTS_DERIVED_SAMPLES + 44 for n in synthetic), 'all120 complete source WAVs')
    native = {n: p for n, p in files.items() if n.startswith('native/')}
    by_id = {}
    for name, facts in native.items():
        parts = base.PurePosixPath(name).parts
        need(len(parts) == 3 and base.re.fullmatch(r'[0-9]{3,}\.s16le', parts[2]), 'ordinary complete native chunks')
        by_id[parts[1]] = by_id.get(parts[1], 0) + facts['bytes']
    need(set(by_id) == {Path(n).stem for n in synthetic} and all(v <= 2 * base.TTS_NATIVE_SAMPLES for v in by_id.values()),
        'all96 native PCM originals within the unchanged full-resample bound')
    payload = wav_names | set(native) | set(measured['members'])
    extra = set(files) - set(value['known_files']) - payload
    need(all(n.endswith('.json') for n in extra) and sum(files[n]['bytes'] for n in extra) <= base.NEW_METADATA_LIMIT,
        'complete new receipts inside the predeclared metadata reserve')
    expanded = sum(p['bytes'] for p in value['known_files'].values()) + base.TTS_AUDIO_BOUND + measured['bytes'] + base.NEW_METADATA_LIMIT
    need(value['maximum_expanded_bytes'] == expanded
        and value['maximum_zip_bytes'] == base.zip_bound(expanded, base.MAX_MEMBERS, base.MAX_MEMBERS * base.MAX_PATH_BYTES),
        'same prospectively calculated complete capacity envelope')
    facts = base._facts(current)
    need(facts['members'] <= base.MAX_MEMBERS and facts['bytes'] <= expanded <= SOURCE_LIMIT
        and base.zip_bound(facts['bytes'], facts['members'], facts['path_bytes']) <= value['maximum_zip_bytes'] <= SOURCE_LIMIT,
        'every whole source byte and ZIP header fits before the next budget')
    return dict(schema=SCHEMA, panel=panel, capacity_only=True, original_wavs=120, files=files,
        actual_bytes=facts['bytes'], members=facts['members'], new_model_calls=0, new_source_calls=0)


def producer_members(directory, panel):
    result = old_capacity.producer_members(directory, panel)
    return dict(result, schema='english-v6-b28-producer-capacity-v1')


def score_budget(rows):
    result = old_capacity.score_budget(rows)
    return dict(result, schema='english-v6-b28-prospective-whole-output-capacity-v1')
