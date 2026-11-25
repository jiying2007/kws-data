"""Prospective B27 byte limits with complete originals kept externally once."""
from __future__ import annotations
from pathlib import Path
import english_v6_b27_profile as profile

base = profile.preserved('english_v6_b24q1_capacity')
need, canonical, sha = profile.need, profile.canonical, profile.sha
SOURCE_LIMIT = profile.LIMITS['source']
CORE_LIMIT = profile.LIMITS['core']
SCHEMA = 'english-v6-b27-prospective-source-capacity-v1'


def budget(directory, corpus, natural_plan, panel):
    directory = Path(directory)
    profile.require_proof(panel)
    observed = base.inventory(directory)['files']
    known = {name: pin for name, pin in observed.items()
             if name not in ('CAPACITY.json', 'CONSUMPTION.json')}
    need({'AUTHORIZATION.json', 'HISTORY.json', 'PROOF_BINDING.json', 'CORPUS_TRANSPORT.json'} <= set(known)
         and any(name.startswith('TRANSPORT/') for name in known)
         and not any(name.startswith(('wav/', 'native/', 'flac/')) for name in known)
         and not any(name.endswith('.zip') for name in known),
         'source metadata retains exact external proof bindings without nested archives')
    natural = base.natural_envelope(corpus, natural_plan)
    expanded = sum(pin['bytes'] for pin in known.values()) + base.TTS_AUDIO_BOUND + natural['bytes'] + base.NEW_METADATA_LIMIT
    packed = base.zip_bound(expanded, base.MAX_MEMBERS, base.MAX_MEMBERS * base.MAX_PATH_BYTES)
    need(expanded <= SOURCE_LIMIT and packed <= SOURCE_LIMIT,
         'STOP before consumption: complete B27 source exceeds prospective 256000000 bytes')
    return dict(schema=SCHEMA, panel=panel, limit_bytes=SOURCE_LIMIT,
        original_artifact_limit_bytes=CORE_LIMIT, external_originals_retained=True,
        proof_binding_sha256=sha((directory / 'PROOF_BINDING.json').read_bytes()),
        known_files=known, natural=natural, natural_plan_sha256=sha(canonical(natural_plan)),
        corpus_sha256_registered=natural_plan['upstream']['sha256'],
        maximum_members=base.MAX_MEMBERS, maximum_path_bytes=base.MAX_PATH_BYTES,
        tts_audio_bytes_max=base.TTS_AUDIO_BOUND, new_metadata_bytes_max=base.NEW_METADATA_LIMIT,
        maximum_expanded_bytes=expanded, maximum_zip_bytes=packed,
        no_member_truncation=True, no_scientific_gate_changed=True,
        new_model_calls=0, new_source_calls=0, new_observer_calls=0)


def before_consumption(directory, corpus, natural_plan, panel):
    directory = Path(directory)
    need(not (directory / 'CONSUMPTION.json').exists()
         and not (directory / 'CAPACITY.json').exists(), 'once-only capacity before scientific consumption')
    return budget(directory, corpus, natural_plan, panel)


def before_generation(directory, corpus, natural_plan, panel):
    expected = budget(directory, corpus, natural_plan, panel)
    need(profile.document((Path(directory) / 'CAPACITY.json').read_bytes()) == expected,
         'whole original capacity and natural headers unchanged before generation')
    return expected


def source_members(directory, panel):
    directory = Path(directory)
    value = profile.document((directory / 'CAPACITY.json').read_bytes())
    need(value['schema'] == SCHEMA and value['panel'] == panel
         and value['limit_bytes'] == SOURCE_LIMIT and value['original_artifact_limit_bytes'] == CORE_LIMIT
         and value['tts_audio_bytes_max'] == base.TTS_AUDIO_BOUND
         and value['new_metadata_bytes_max'] == base.NEW_METADATA_LIMIT,
         'exact prospective B27 envelope; no later cap expansion')
    current = base.inventory(directory)
    files = current['files']
    need(all(files.get(name) == pin for name, pin in value['known_files'].items())
         and {'AUTHORIZATION.json', 'HISTORY.json', 'PROOF_BINDING.json', 'CORPUS_TRANSPORT.json'} <= set(value['known_files'])
         and value['proof_binding_sha256'] == sha((directory / 'PROOF_BINDING.json').read_bytes()),
         'all complete original pre-consumption metadata retained byte-for-byte')
    need(not any(name.endswith('.zip') for name in files), 'historical archives remain external and complete')
    natural_rows = {row["source_id"]: row for row in value["natural"]["members"].values()}
    need(len(natural_rows) == base.NATURAL_COUNT
         and all(type(row["samples"]) is int and row["samples"] > 0
                 and type(row["flac_bytes"]) is int and row["flac_bytes"] >= 42
                 and row["wav_bytes"] == 2 * row["samples"] + 44 for row in natural_rows.values())
         and value["natural"]["bytes"] == sum(row["flac_bytes"] + row["wav_bytes"] for row in natural_rows.values()),
         "all original natural slots and exact whole-PCM arithmetic in saved budget")
    wav_names = {name for name in files if name.startswith("wav/")}
    flac_names = {name for name in files if name.startswith("flac/")}
    natural_wavs = set()
    for ident, row in natural_rows.items():
        stem = "b27-" + panel + "-natural-" + ident
        wav, flac = "wav/" + stem + ".wav", "flac/" + stem + ".flac"
        need(files.get(wav, {}).get("bytes") == row["wav_bytes"]
             and files.get(flac, {}).get("bytes") == row["flac_bytes"], "whole natural FLAC and PCM lengths, including rejects")
        natural_wavs.add(wav)
    need(len(flac_names) == base.NATURAL_COUNT and len(wav_names) == base.TTS_COUNT + base.NATURAL_COUNT,
         "all and only 120 complete source WAVs and 24 original FLACs")
    synthetic_wavs = wav_names - natural_wavs
    need(len(synthetic_wavs) == base.TTS_COUNT
         and all(files[name]["bytes"] <= 2 * base.TTS_DERIVED_SAMPLES + 44 for name in synthetic_wavs),
         "96 bounded whole synthetic PCM originals")
    native = {name: facts for name, facts in files.items() if name.startswith("native/")}
    native_by_id = {}
    for name, facts in native.items():
        parts = base.PurePosixPath(name).parts
        need(len(parts) == 3 and base.re.fullmatch(r"[0-9]{3,}\.s16le", parts[2]), "all original native chunks")
        native_by_id[parts[1]] = native_by_id.get(parts[1], 0) + facts["bytes"]
    need(set(native_by_id) == {base.PurePosixPath(name).stem for name in synthetic_wavs}
         and all(size <= 2 * base.TTS_NATIVE_SAMPLES for size in native_by_id.values()),
         "all 96 original native PCM outputs fit the complete-resample bound")

    reserved = set(value['known_files'])
    payload = wav_names | flac_names | set(native)
    metadata = set(files) - reserved - payload
    need(all(name.endswith('.json') for name in metadata)
         and sum(files[name]['bytes'] for name in metadata) <= base.NEW_METADATA_LIMIT,
         'all new original metadata fits the before-call reserve')
    expanded = sum(pin['bytes'] for pin in value['known_files'].values()) + base.TTS_AUDIO_BOUND + value['natural']['bytes'] + base.NEW_METADATA_LIMIT
    need(value['maximum_expanded_bytes'] == expanded
         and value['maximum_zip_bytes'] == base.zip_bound(expanded, base.MAX_MEMBERS, base.MAX_MEMBERS * base.MAX_PATH_BYTES),
         'exact prospectively computed complete source byte envelope')
    facts = base._facts(current)
    need(facts['members'] <= base.MAX_MEMBERS and facts['bytes'] <= expanded <= SOURCE_LIMIT
         and base.zip_bound(facts['bytes'], facts['members'], facts['path_bytes']) <= value['maximum_zip_bytes'] <= SOURCE_LIMIT,
         'all original source bytes, names and headers fit before any next budget')
    return dict(schema=SCHEMA, panel=panel, capacity_only=True, original_wavs=120, files=files,
                actual_bytes=facts['bytes'], members=facts['members'], new_model_calls=0, new_source_calls=0)


def producer_members(directory, panel):
    need(panel in ('primary', 'replication'), 'explicit original B27 scoring producer')
    directory = Path(directory)
    pieces = []
    for name in (panel + '_producer', 'source_producer'):
        value = base.inventory(directory / name)
        need(value['files'], 'complete nonempty original code producer')
        pieces.append(base._facts(value, name + '/'))
    for name in ('FETCH.json', 'CONSUMPTION.json'):
        path = directory / name
        size = base.CONSUMPTION_RESERVE if name == 'CONSUMPTION.json' and not path.exists() else path.stat().st_size
        need(name != 'CONSUMPTION.json' or size <= base.CONSUMPTION_RESERVE, 'bounded actual consumption marker')
        pieces.append(dict(bytes=size, members=1, path_bytes=len(name)))
    total = {key: sum(item[key] for item in pieces) for key in ('bytes', 'members', 'path_bytes')}
    total['members'] += 2
    total['path_bytes'] += len(panel + '_producer/') + len('source_producer/')
    bound = base.zip_bound(total['bytes'], total['members'], total['path_bytes'])
    need(total['members'] <= base.MAX_MEMBERS and total['bytes'] <= CORE_LIMIT and bound <= CORE_LIMIT,
         'complete original producer fits unchanged128MB before scoring consumption')
    return dict(schema='english-v6-b27-producer-capacity-v1', panel=panel, **total,
                maximum_zip_bytes=bound, limit_bytes=CORE_LIMIT, historical_archives_nested=0)


def score_budget(rows):
    need(type(rows) is list and 0 < len(rows) <= 120
         and all(type(row['samples']) is int and 3200 <= row['samples'] <= 240000 for row in rows),
         'every whole admitted source before the output budget')
    count = len(rows) * 12
    frames = sum((row['samples'] - 512) // 160 + 1 for row in rows) * 12
    # Every actual FP32 batch plus its complete matrix, NPY headers, journals,
    # every metadata object and ZIP overhead are covered before first forward.
    metadata_reserve = 16_000_000
    expanded = frames * 4 * 4 * 2 + count * 512 * 2 + metadata_reserve
    members = 3 * count + 16
    bound = base.zip_bound(expanded, members, members * 512)
    need(count <= 1440 and members <= base.MAX_MEMBERS and expanded <= CORE_LIMIT and bound <= CORE_LIMIT,
         'STOP before scoring consumption: full raw-output package exceeds original128MB limit')
    return dict(schema='english-v6-b27-prospective-whole-output-capacity-v1', arrays=count,
        whole_original_frames=frames, full_fp32_bytes=frames * 16,
        metadata_reserve=metadata_reserve, maximum_expanded_bytes=expanded, maximum_zip_bytes=bound,
        limit_bytes=CORE_LIMIT, all_raw_returns_and_matrices_retained=True,
        new_model_calls=0, retrospective_cap_change_allowed=False)
